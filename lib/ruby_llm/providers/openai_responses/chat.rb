# frozen_string_literal: true

module RubyLLM
  module Providers
    class OpenAIResponses
      # Chat completion methods for the OpenAI Responses API.
      # Handles converting RubyLLM messages to Responses API format and parsing responses.
      module Chat
        def completion_url
          'responses'
        end

        module_function

        # rubocop:disable Metrics/ParameterLists
        def render_payload(messages, tools:, temperature:, model:, stream: false,
                           schema: nil, thinking: nil, tool_prefs: nil) # rubocop:disable Lint/UnusedMethodArgument
          tool_prefs ||= {}
          non_system_messages = messages.reject { |m| m.role == :system }
          continuation_input = continuation_input_messages(non_system_messages)

          payload = {
            model: model.id,
            input: format_input(continuation_input || messages),
            stream: stream
          }

          payload[:temperature] = temperature unless temperature.nil?
          apply_tools(payload, tools, tool_prefs)
          payload[:text] = build_schema_format(schema) if schema

          last_response_id = extract_last_response_id(messages)
          payload[:previous_response_id] = last_response_id if last_response_id

          payload
        end
        # rubocop:enable Metrics/ParameterLists

        # Apply tools and tool preferences to the payload.
        def apply_tools(payload, tools, tool_prefs)
          return unless tools.any?

          payload[:tools] = tools.map { |_, tool| Tools.tool_for(tool) }
          payload[:tool_choice] = build_tool_choice(tool_prefs[:choice]) unless tool_prefs[:choice].nil?
          payload[:parallel_tool_calls] = tool_prefs[:calls] == :many unless tool_prefs[:calls].nil?
        end

        # Convert a RubyLLM tool choice symbol to the Responses API format.
        # Responses API accepts "auto", "required", "none", or
        # { type: "function", name: "fn_name" } for a specific function.
        def build_tool_choice(choice)
          case choice
          when :auto, :none, :required
            choice.to_s
          else
            { type: 'function', name: choice.to_s }
          end
        end

        # Build the Responses API text format block from a schema.
        # Schema arrives pre-normalized as { name:, schema:, strict: } from
        # RubyLLM::Chat.with_schema (v1.13+), or as a raw hash (legacy).
        def build_schema_format(schema)
          schema_name = schema[:name] || 'response'
          schema_def = schema[:schema] || schema
          strict = schema.key?(:strict) ? schema[:strict] : true

          {
            format: {
              type: 'json_schema',
              name: schema_name,
              schema: schema_def,
              strict: strict
            }
          }
        end

        def extract_last_response_id(messages)
          messages
            .select { |m| m.role == :assistant && m.respond_to?(:response_id) }
            .map(&:response_id)
            .compact
            .last
        end

        def continuation_input_messages(messages)
          last_response_index = messages.rindex do |message|
            message.role == :assistant && message.respond_to?(:response_id) && message.response_id
          end
          return nil unless last_response_index

          trailing_messages = messages[(last_response_index + 1)..]
          return nil unless trailing_messages&.any?
          return nil unless trailing_messages.all? { |message| message.role == :tool }

          trailing_messages
        end

        def parse_completion_response(response)
          data = response.body
          return if data.nil? || data.empty?

          data = JSON.parse(data) if data.is_a?(String)

          raise RubyLLM::Error.new(response, data.dig('error', 'message')) if data.dig('error', 'message')

          output = data['output'] || []

          # Extract text content from output
          content = extract_output_text(output)

          # Extract executable tool calls from function_call and local shell_call outputs
          tool_calls = extract_tool_calls(output, response_tools: data['tools'])

          usage = data['usage'] || {}
          cached_tokens = usage.dig('input_tokens_details', 'cached_tokens')

          Message.new(
            role: :assistant,
            content: content,
            tool_calls: tool_calls,
            input_tokens: usage['input_tokens'],
            output_tokens: usage['output_tokens'],
            cached_tokens: cached_tokens,
            cache_creation_tokens: 0,
            model_id: data['model'],
            response_id: data['id'],
            raw: response
          )
        end

        def format_input(messages)
          result = []

          messages.each do |msg|
            if msg.tool_call_id
              result << format_tool_result(msg)
            elsif msg.tool_calls&.any?
              # Assistant message with tool calls
              # First add any text content as a message
              text = extract_text_content(msg.content)
              if text && !text.empty?
                result << {
                  type: 'message',
                  role: 'assistant',
                  content: text
                }
              end

              # Then add each function call as a separate item
              msg.tool_calls.each_value do |tc|
                result << format_tool_call(tc)
              end
            else
              # Regular message
              result << {
                type: 'message',
                role: format_role(msg.role),
                content: format_message_content(msg.content, nil)
              }
            end
          end

          result
        end

        def format_tool_result(msg)
          content = msg.content
          return content.value if LocalShellExecutor.shell_call_output?(content)

          {
            type: 'function_call_output',
            call_id: msg.tool_call_id,
            output: format_function_tool_output(content)
          }
        end

        def format_function_tool_output(content)
          return raw_tool_output(content.value) if content.is_a?(RubyLLM::Content::Raw)

          extract_text_content(content)
        end

        def raw_tool_output(value)
          value.is_a?(String) ? value : JSON.generate(value)
        end

        def format_tool_call(tool_call)
          return tool_call.shell_call if tool_call.is_a?(LocalShellToolCall)

          {
            type: 'function_call',
            call_id: tool_call.id,
            name: tool_call.name,
            arguments: tool_call.arguments.is_a?(String) ? tool_call.arguments : JSON.generate(tool_call.arguments)
          }
        end

        def format_message_content(content, tool_calls = nil)
          parts = []

          # Add text content
          text = extract_text_content(content)
          parts << { type: 'input_text', text: text } if text && !text.empty?

          # Add attachments if present
          if content.is_a?(RubyLLM::Content)
            content.attachments.each do |attachment|
              parts << format_attachment(attachment)
            end
          end

          # Add tool calls if present (for assistant messages)
          if tool_calls&.any?
            tool_calls.each_value do |tc|
              parts << format_tool_call(tc)
            end
          end

          # Return simple text for single text content
          return parts.first[:text] if parts.length == 1 && parts.first[:type] == 'input_text'

          parts
        end

        def format_attachment(attachment)
          case attachment.type
          when :image
            if attachment.url?
              { type: 'input_image', image_url: attachment.source }
            else
              { type: 'input_image', image_url: attachment.for_llm }
            end
          when :pdf
            {
              type: 'input_file',
              filename: File.basename(attachment.source.to_s),
              file_data: attachment.for_llm
            }
          when :audio
            {
              type: 'input_audio',
              data: attachment.for_llm,
              format: detect_audio_format(attachment.source)
            }
          else
            { type: 'input_text', text: "[Unsupported attachment: #{attachment.type}]" }
          end
        end

        def detect_audio_format(source)
          ext = File.extname(source.to_s).downcase
          case ext
          when '.mp3' then 'mp3'
          when '.wav' then 'wav'
          when '.webm' then 'webm'
          when '.ogg' then 'ogg'
          when '.flac' then 'flac'
          else 'mp3'
          end
        end

        def extract_text_content(content)
          case content
          when String
            content
          when RubyLLM::Content
            content.text
          when Hash
            content[:text] || content['text']
          else
            content.to_s
          end
        end

        def format_role(role)
          case role
          when :system then 'developer'
          when :assistant then 'assistant'
          when :tool then 'user' # Tool results come from user perspective
          else role.to_s
          end
        end

        def extract_output_text(output)
          output
            .select { |item| item['type'] == 'message' }
            .flat_map { |item| item['content'] || [] }
            .select { |c| c['type'] == 'output_text' }
            .map { |c| c['text'] }
            .join
        end

        def extract_tool_calls(output, response_tools: nil)
          executable_calls = output.select do |item|
            item['type'] == 'function_call' || local_shell_call?(item, response_tools: response_tools)
          end
          return nil if executable_calls.empty?

          executable_calls.to_h do |item|
            tool_call = if local_shell_call?(item, response_tools: response_tools)
                          LocalShellToolCall.new(item)
                        else
                          ToolCall.new(
                            id: item['call_id'],
                            name: item['name'],
                            arguments: parse_arguments(item['arguments'])
                          )
                        end
            [tool_call.id, tool_call]
          end
        end

        def local_shell_call?(item, response_tools: nil)
          return false unless item['type'] == 'shell_call'
          return true if item.dig('environment', 'type') == 'local'

          item['environment'].nil? && local_shell_tool_configured?(response_tools)
        end

        def local_shell_tool_configured?(tools)
          Array(tools).any? { |tool| tool['type'] == 'shell' && tool.dig('environment', 'type') == 'local' }
        end

        def parse_arguments(arguments)
          return {} if arguments.nil? || arguments.empty?
          return arguments if arguments.is_a?(Hash)

          JSON.parse(arguments)
        rescue JSON::ParserError
          { raw: arguments }
        end
      end
    end
  end
end
