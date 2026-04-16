# frozen_string_literal: true

require 'spec_helper'

class LocalShellLifecycleEchoTool < RubyLLM::Tool
  description 'Echo a value'
  param :value, type: 'string', desc: 'Value to echo'

  def execute(value:)
    value
  end
end

RSpec.describe 'OpenAI Responses local shell lifecycle' do
  let(:endpoint) { 'https://api.openai.com/v1/responses' }
  let(:model) { 'gpt-5.4' }
  let(:shell_tool) { RubyLLM::ResponsesAPI::BuiltInTools.shell(environment_type: 'local') }

  def build_chat
    RubyLLM.chat(model: model, provider: :openai_responses, assume_model_exists: true)
  end

  def stub_responses(*bodies)
    requests = []
    response_index = 0

    stub_request(:post, endpoint).to_return do |request|
      requests << JSON.parse(request.body)
      body = bodies.fetch(response_index)
      response_index += 1
      {
        status: 200,
        body: JSON.generate(body),
        headers: { 'Content-Type' => 'application/json' }
      }
    end

    requests
  end

  def stub_streaming_responses(*event_groups)
    requests = []
    response_index = 0

    stub_request(:post, endpoint).to_return do |request|
      requests << JSON.parse(request.body)
      events = event_groups.fetch(response_index)
      response_index += 1
      {
        status: 200,
        body: build_sse_body(events),
        headers: { 'Content-Type' => 'text/event-stream' }
      }
    end

    requests
  end

  def shell_call_response(environment: { 'type' => 'local' })
    {
      'id' => 'resp_shell_1',
      'model' => model,
      'output' => [
        {
          'type' => 'shell_call',
          'id' => 'sh_1',
          'call_id' => 'call_shell_1',
          'status' => 'in_progress',
          'environment' => environment,
          'action' => {
            'commands' => ['pwd'],
            'timeout_ms' => 10_000,
            'max_output_length' => 2_000
          }
        }
      ],
      'tools' => [shell_tool],
      'usage' => { 'input_tokens' => 12, 'output_tokens' => 6 }
    }
  end

  def final_response
    {
      'id' => 'resp_final_1',
      'model' => model,
      'output' => [
        {
          'type' => 'message',
          'role' => 'assistant',
          'content' => [{ 'type' => 'output_text', 'text' => 'Done from shell.' }]
        }
      ],
      'usage' => { 'input_tokens' => 8, 'output_tokens' => 4 }
    }
  end

  def completed_event(response)
    {
      'type' => 'response.completed',
      'response' => response
    }
  end

  def final_response_stream_events
    [
      { 'type' => 'response.output_text.delta', 'delta' => 'Done ' },
      { 'type' => 'response.output_text.delta', 'delta' => 'from shell.' },
      completed_event(final_response)
    ]
  end

  it 'executes local shell calls and continues with only shell_call_output' do
    requests = stub_responses(shell_call_response, final_response)
    executor_calls = []
    tool_calls = []
    tool_results = []
    end_messages = []

    local_shell_executor = lambda do |shell_call|
      executor_calls << shell_call
      [
        {
          'stdout' => "/repo\n",
          'stderr' => '',
          'outcome' => { 'type' => 'exit', 'exit_code' => 0 }
        }
      ]
    end

    chat = build_chat
    chat.with_instructions('Be brief.')
    chat.with_params(tools: [shell_tool], local_shell_executor: local_shell_executor)
    chat.on_tool_call { |tool_call| tool_calls << tool_call }
    chat.on_tool_result { |result| tool_results << result }
    chat.on_end_message { |message| end_messages << message }

    response = chat.ask('Inspect the repo')

    expect(response.content).to eq('Done from shell.')
    expect(executor_calls.length).to eq(1)
    expect(executor_calls.first['call_id']).to eq('call_shell_1')
    expect(tool_calls.first).to be_a(RubyLLM::Providers::OpenAIResponses::LocalShellToolCall)
    expect(tool_results.first).to be_a(RubyLLM::Content::Raw)
    expect(end_messages.map(&:role)).to eq(%i[assistant tool assistant])

    expect(requests.length).to eq(2)
    expect(requests.first['input'].first['content']).to eq('Inspect the repo')
    expect(requests.first['instructions']).to eq('Be brief.')
    expect(requests.first).not_to have_key('local_shell_executor')

    continuation = requests.last
    expect(continuation['previous_response_id']).to eq('resp_shell_1')
    expect(continuation).not_to have_key('instructions')
    expect(continuation).not_to have_key('local_shell_executor')
    expect(continuation['input']).to eq(
      [
        {
          'type' => 'shell_call_output',
          'call_id' => 'call_shell_1',
          'max_output_length' => 2_000,
          'output' => [
            {
              'stdout' => "/repo\n",
              'stderr' => '',
              'outcome' => { 'type' => 'exit', 'exit_code' => 0 }
            }
          ]
        }
      ]
    )
  end

  it 'streams final assistant content after executing a local shell call' do
    requests = stub_streaming_responses(
      [completed_event(shell_call_response(environment: nil))],
      final_response_stream_events
    )
    executor_calls = []
    streamed_content = []

    chat = build_chat
    chat.with_params(
      tools: [shell_tool],
      local_shell_executor: lambda do |shell_call|
        executor_calls << shell_call
        [{ 'stdout' => "/repo\n", 'stderr' => '', 'outcome' => { 'type' => 'exit', 'exit_code' => 0 } }]
      end
    )

    response = chat.ask('Inspect the repo') do |chunk|
      streamed_content << chunk.content if chunk.content
    end

    expect(response.content).to eq('Done from shell.')
    expect(streamed_content).to eq(['Done ', 'from shell.'])
    expect(executor_calls.length).to eq(1)
    expect(requests.length).to eq(2)
    expect(requests.map { |request| request['stream'] }).to eq([true, true])
    expect(requests.last['previous_response_id']).to eq('resp_shell_1')
    expect(requests.last['input'].first['type']).to eq('shell_call_output')
  end

  it 'fails clearly when a local shell call has no executor' do
    stub_responses(shell_call_response)

    chat = build_chat
    chat.with_params(tools: [shell_tool])

    expect { chat.ask('Inspect the repo') }
      .to raise_error(RubyLLM::Error, /local_shell_executor/)
  end

  it 'keeps string-keyed local shell executor params local-only' do
    requests = stub_responses(shell_call_response, final_response)
    executor_calls = []

    chat = build_chat
    chat.with_params(
      **{
        tools: [shell_tool],
        'local_shell_executor' => lambda do |shell_call|
          executor_calls << shell_call
          [{ 'stdout' => "/repo\n", 'stderr' => '', 'outcome' => { 'type' => 'exit', 'exit_code' => 0 } }]
        end
      }
    )

    response = chat.ask('Inspect the repo')

    expect(response.content).to eq('Done from shell.')
    expect(executor_calls.length).to eq(1)
    expect(requests.length).to eq(2)
    expect(requests.first).not_to have_key('local_shell_executor')
    expect(requests.last).not_to have_key('local_shell_executor')
  end

  it 'uses shell call id as the output call_id when call_id is missing' do
    shell_response = shell_call_response
    shell_response['output'].first.delete('call_id')
    requests = stub_responses(shell_response, final_response)

    chat = build_chat
    chat.with_params(
      tools: [shell_tool],
      local_shell_executor: lambda do |_shell_call|
        [{ 'stdout' => "/repo\n", 'stderr' => '', 'outcome' => { 'type' => 'exit', 'exit_code' => 0 } }]
      end
    )

    response = chat.ask('Inspect the repo')

    expect(response.content).to eq('Done from shell.')
    expect(requests.last['input'].first['type']).to eq('shell_call_output')
    expect(requests.last['input'].first['call_id']).to eq('sh_1')
  end

  it 'leaves hosted shell calls to existing built-in tool behavior' do
    hosted_shell_tool = RubyLLM::ResponsesAPI::BuiltInTools.shell
    response = shell_call_response(environment: { 'type' => 'container_auto' }).merge('tools' => [hosted_shell_tool])
    requests = stub_responses(response)
    executor_calls = []

    chat = build_chat
    chat.with_params(
      tools: [RubyLLM::ResponsesAPI::BuiltInTools.shell],
      local_shell_executor: ->(shell_call) { executor_calls << shell_call }
    )

    response = chat.ask('Inspect the repo')

    expect(response.tool_call?).to be false
    expect(executor_calls).to be_empty
    expect(requests.length).to eq(1)
  end

  it 'executes shell calls with nil environment when the response declares a local shell tool' do
    response = shell_call_response(environment: nil)
    requests = stub_responses(response, final_response)
    executor_calls = []

    chat = build_chat
    chat.with_params(
      tools: [shell_tool],
      local_shell_executor: lambda do |shell_call|
        executor_calls << shell_call
        [{ 'stdout' => "/repo\n", 'stderr' => '', 'outcome' => { 'type' => 'exit', 'exit_code' => 0 } }]
      end
    )

    final_message = chat.ask('Inspect the repo')

    expect(final_message.content).to eq('Done from shell.')
    expect(executor_calls.length).to eq(1)
    expect(executor_calls.first['environment']).to be_nil
    expect(requests.last['previous_response_id']).to eq('resp_shell_1')
    expect(requests.last['input'].first['type']).to eq('shell_call_output')
  end

  it 'leaves nil-environment shell calls alone when the response does not declare a local shell tool' do
    hosted_shell_tool = RubyLLM::ResponsesAPI::BuiltInTools.shell
    response = shell_call_response(environment: nil).merge('tools' => [hosted_shell_tool])
    requests = stub_responses(response)
    executor_calls = []

    chat = build_chat
    chat.with_params(
      tools: [hosted_shell_tool],
      local_shell_executor: ->(shell_call) { executor_calls << shell_call }
    )

    response_message = chat.ask('Inspect the repo')

    expect(response_message.tool_call?).to be false
    expect(executor_calls).to be_empty
    expect(requests.length).to eq(1)
  end

  it 'rejects non-array executor results' do
    stub_responses(shell_call_response)

    chat = build_chat
    chat.with_params(
      tools: [shell_tool],
      local_shell_executor: ->(_shell_call) { { 'stdout' => 'ok' } }
    )

    expect { chat.ask('Inspect the repo') }
      .to raise_error(RubyLLM::Error, /array of command result hashes/)
  end

  it 'mixes function tool calls and local shell calls in the same continuation' do
    first_response = shell_call_response.merge(
      'output' => [
        {
          'type' => 'function_call',
          'call_id' => 'call_echo_1',
          'name' => 'local_shell_lifecycle_echo',
          'arguments' => '{"value":"hello"}'
        },
        shell_call_response['output'].first
      ]
    )
    requests = stub_responses(first_response, final_response)

    chat = build_chat
    chat.with_tool(LocalShellLifecycleEchoTool)
    function_tool = RubyLLM::Providers::OpenAIResponses::Tools.tool_for(chat.tools.fetch(:local_shell_lifecycle_echo))
    chat.with_params(
      tools: [function_tool, shell_tool],
      local_shell_executor: lambda do |_shell_call|
        [{ 'stdout' => 'ok', 'stderr' => '', 'outcome' => { 'type' => 'exit', 'exit_code' => 0 } }]
      end
    )

    response = chat.ask('Use both tools')

    expect(response.content).to eq('Done from shell.')
    expect(requests.last['previous_response_id']).to eq('resp_shell_1')
    expect(requests.last['input'].map { |item| item['type'] }).to eq(%w[function_call_output shell_call_output])
    expect(requests.last['input'].first['output']).to eq('hello')
    expect(requests.last['input'].last['call_id']).to eq('call_shell_1')
  end

  it 'continues function-only tool calls with incremental input' do
    first_response = {
      'id' => 'resp_function_1',
      'model' => model,
      'output' => [
        {
          'type' => 'function_call',
          'call_id' => 'call_echo_1',
          'name' => 'local_shell_lifecycle_echo',
          'arguments' => '{"value":"hello"}'
        }
      ],
      'usage' => { 'input_tokens' => 12, 'output_tokens' => 6 }
    }
    requests = stub_responses(first_response, final_response)

    chat = build_chat
    chat.with_tool(LocalShellLifecycleEchoTool)
    function_tool = RubyLLM::Providers::OpenAIResponses::Tools.tool_for(chat.tools.fetch(:local_shell_lifecycle_echo))
    chat.with_params(
      tools: [function_tool, shell_tool],
      local_shell_executor: lambda do |_shell_call|
        [{ 'stdout' => 'ok', 'stderr' => '', 'outcome' => { 'type' => 'exit', 'exit_code' => 0 } }]
      end
    )

    response = chat.ask('Use the function tool')

    expect(response.content).to eq('Done from shell.')
    expect(requests.last['previous_response_id']).to eq('resp_function_1')
    expect(requests.last).not_to have_key('instructions')
    expect(requests.last['input']).to eq(
      [
        {
          'type' => 'function_call_output',
          'call_id' => 'call_echo_1',
          'output' => 'hello'
        }
      ]
    )
  end
end
