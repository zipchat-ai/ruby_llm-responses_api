# frozen_string_literal: true

module RubyLLM
  module Providers
    class OpenAIResponses
      # Executes Responses API local shell calls through RubyLLM's normal tool lifecycle.
      class LocalShellExecutor
        TOOL_NAME = 'openai_responses_local_shell'

        attr_reader :executor

        def initialize(executor = nil, &block)
          @executor = executor || block
          return if @executor.nil? || @executor.respond_to?(:call)

          raise ArgumentError, 'local_shell_executor must respond to #call.'
        end

        def name
          TOOL_NAME
        end

        def description
          'Execute an OpenAI Responses API local shell call.'
        end

        def parameters
          []
        end

        def params_schema
          {
            'type' => 'object',
            'properties' => {},
            'additionalProperties' => true
          }
        end

        def provider_params
          {}
        end

        def call(shell_call)
          raise RubyLLM::Error, 'OpenAI Responses local shell call requires a local_shell_executor.' unless executor

          RubyLLM::Content::Raw.new(normalize_output(shell_call, executor.call(shell_call)))
        end

        def self.local_shell_tool_call?(tool_call)
          tool_call.name == TOOL_NAME
        end

        def self.shell_call_output?(content)
          content.is_a?(RubyLLM::Content::Raw) && content.value.is_a?(Hash) &&
            content.value['type'] == 'shell_call_output'
        end

        def self.shell_call_id(shell_call)
          shell_call['call_id'] || shell_call['id']
        end

        private

        def normalize_output(shell_call, result)
          normalized = wrap_output(shell_call, normalize_command_results(result))
          validate_output!(normalized)
          normalized
        end

        def normalize_command_results(result)
          unless result.is_a?(Array)
            raise RubyLLM::Error, 'local_shell_executor must return an array of command result hashes.'
          end

          result.map do |item|
            raise RubyLLM::Error, 'local_shell_executor command results must be hashes.' unless item.is_a?(Hash)

            stringify_keys(item)
          end
        end

        def wrap_output(shell_call, command_results)
          action = shell_call['action'] || {}
          {
            'type' => 'shell_call_output',
            'call_id' => self.class.shell_call_id(shell_call),
            'max_output_length' => action['max_output_length'],
            'output' => command_results
          }.compact
        end

        def validate_output!(output)
          raise RubyLLM::Error, 'local shell calls must include a call_id.' if blank?(output['call_id'])

          return if output['output'].is_a?(Array)

          raise RubyLLM::Error, 'local shell output must include an output array.'
        end

        def blank?(value)
          value.nil? || (value.respond_to?(:empty?) && value.empty?)
        end

        def stringify_keys(value)
          case value
          when Hash
            value.each_with_object({}) do |(key, val), result|
              result[key.to_s] = stringify_keys(val)
            end
          when Array
            value.map { |item| stringify_keys(item) }
          else
            value
          end
        end
      end

      # ToolCall subtype carrying the original Responses API shell_call item.
      class LocalShellToolCall < RubyLLM::ToolCall
        attr_reader :shell_call

        def initialize(shell_call)
          @shell_call = shell_call
          super(
            id: LocalShellExecutor.shell_call_id(shell_call),
            name: LocalShellExecutor::TOOL_NAME,
            arguments: shell_call
          )
        end
      end

      # Lets with_params(local_shell_executor: ...) execute local shell calls without
      # exposing the executor as a function tool in the model request.
      module ChatExtension
        def execute_tool(tool_call)
          return super unless LocalShellExecutor.local_shell_tool_call?(tool_call)

          registered_tool = tools[tool_call.name.to_sym]
          return registered_tool.call(tool_call.arguments) if registered_tool

          executor = @params[:local_shell_executor] || @params['local_shell_executor']
          LocalShellExecutor.new(executor).call(tool_call.arguments)
        end

        private :execute_tool
      end
    end
  end
end

RubyLLM::Chat.prepend(RubyLLM::Providers::OpenAIResponses::ChatExtension)
