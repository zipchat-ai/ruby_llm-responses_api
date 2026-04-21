# frozen_string_literal: true

require 'spec_helper'

RSpec.describe RubyLLM::Providers::OpenAIResponses::Chat do
  let(:chat_module) { RubyLLM::Providers::OpenAIResponses::Chat }
  let(:model) { instance_double(RubyLLM::Model::Info, id: 'gpt-4o') }

  describe '.render_payload' do
    let(:user_message) do
      RubyLLM::Message.new(role: :user, content: 'Hello')
    end

    let(:system_message) do
      RubyLLM::Message.new(role: :system, content: 'You are a helpful assistant')
    end

    it 'creates basic payload with model and input' do
      payload = chat_module.render_payload(
        [user_message],
        tools: {},
        temperature: nil,
        model: model,
        stream: false
      )

      expect(payload[:model]).to eq('gpt-4o')
      expect(payload[:input]).to be_an(Array)
      expect(payload[:stream]).to be false
    end

    it 'formats system messages as developer input messages' do
      payload = chat_module.render_payload(
        [system_message, user_message],
        tools: {},
        temperature: nil,
        model: model,
        stream: false
      )

      expect(payload).not_to have_key(:instructions)
      expect(payload[:input]).to eq(
        [
          { type: 'message', role: 'developer', content: 'You are a helpful assistant' },
          { type: 'message', role: 'user', content: 'Hello' }
        ]
      )
    end

    it 'preserves multiple system messages as developer input messages' do
      second_system_message = RubyLLM::Message.new(role: :system, content: 'Always be concise')

      payload = chat_module.render_payload(
        [system_message, second_system_message, user_message],
        tools: {},
        temperature: nil,
        model: model,
        stream: false
      )

      expect(payload).not_to have_key(:instructions)
      expect(payload[:input].map { |item| item[:role] }).to eq(%w[developer developer user])
      expect(payload[:input].map { |item| item[:content] }).to eq(
        ['You are a helpful assistant', 'Always be concise', 'Hello']
      )
    end

    it 'includes temperature when provided' do
      payload = chat_module.render_payload(
        [user_message],
        tools: {},
        temperature: 0.7,
        model: model,
        stream: false
      )

      expect(payload[:temperature]).to eq(0.7)
    end

    context 'with tool_prefs' do
      let(:tool) do
        instance_double(RubyLLM::Tool, name: 'get_weather', description: 'Get weather',
                                       parameters: [], params_schema: nil)
      end
      let(:tools) { { get_weather: tool } }

      before do
        allow(RubyLLM::Providers::OpenAIResponses::Tools)
          .to receive(:tool_for).and_return({ type: 'function', name: 'get_weather' })
      end

      it 'includes tool_choice when choice is set' do
        payload = chat_module.render_payload(
          [user_message], tools: tools, temperature: nil, model: model,
                          tool_prefs: { choice: :required, calls: nil }
        )

        expect(payload[:tool_choice]).to eq('required')
      end

      it 'includes parallel_tool_calls when calls is :many' do
        payload = chat_module.render_payload(
          [user_message], tools: tools, temperature: nil, model: model,
                          tool_prefs: { choice: nil, calls: :many }
        )

        expect(payload[:parallel_tool_calls]).to be true
      end

      it 'sets parallel_tool_calls false when calls is :one' do
        payload = chat_module.render_payload(
          [user_message], tools: tools, temperature: nil, model: model,
                          tool_prefs: { choice: nil, calls: :one }
        )

        expect(payload[:parallel_tool_calls]).to be false
      end

      it 'formats specific function choice correctly' do
        payload = chat_module.render_payload(
          [user_message], tools: tools, temperature: nil, model: model,
                          tool_prefs: { choice: :get_weather, calls: nil }
        )

        expect(payload[:tool_choice]).to eq({ type: 'function', name: 'get_weather' })
      end

      it 'omits tool_choice and parallel_tool_calls when nil' do
        payload = chat_module.render_payload(
          [user_message], tools: tools, temperature: nil, model: model,
                          tool_prefs: { choice: nil, calls: nil }
        )

        expect(payload).not_to have_key(:tool_choice)
        expect(payload).not_to have_key(:parallel_tool_calls)
      end
    end
  end

  describe '.parse_completion_response' do
    it 'parses a successful response' do
      response = mock_response(sample_completion_response)
      message = chat_module.parse_completion_response(response)

      expect(message.role).to eq(:assistant)
      expect(message.content).to eq('Hello! How can I help you today?')
      expect(message.input_tokens).to eq(10)
      expect(message.output_tokens).to eq(8)
    end

    it 'extracts tool calls from function_call outputs' do
      response = mock_response(sample_tool_call_response)
      message = chat_module.parse_completion_response(response)

      expect(message.tool_calls).to be_a(Hash)
      expect(message.tool_calls['call_abc123'].name).to eq('get_weather')
      expect(message.tool_calls['call_abc123'].arguments).to eq({ 'location' => 'San Francisco' })
    end
  end

  describe '.format_input' do
    it 'formats user messages correctly' do
      messages = [RubyLLM::Message.new(role: :user, content: 'Test message')]
      input = chat_module.format_input(messages)

      expect(input.first[:type]).to eq('message')
      expect(input.first[:role]).to eq('user')
      expect(input.first[:content]).to eq('Test message')
    end

    it 'formats tool result messages as function_call_output' do
      messages = [
        RubyLLM::Message.new(
          role: :tool,
          content: '{"result": "success"}',
          tool_call_id: 'call_123'
        )
      ]
      input = chat_module.format_input(messages)

      expect(input.first[:type]).to eq('function_call_output')
      expect(input.first[:call_id]).to eq('call_123')
      expect(input.first[:output]).to eq('{"result": "success"}')
    end

    it 'serializes non-shell raw tool results as function output strings' do
      messages = [
        RubyLLM::Message.new(
          role: :tool,
          content: RubyLLM::Content::Raw.new({ 'result' => ['success'] }),
          tool_call_id: 'call_123'
        )
      ]

      input = chat_module.format_input(messages)

      expect(input.first[:type]).to eq('function_call_output')
      expect(input.first[:call_id]).to eq('call_123')
      expect(input.first[:output]).to eq('{"result":["success"]}')
    end
  end
end
