import os
from dotenv import load_dotenv, find_dotenv
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    Runner,
    function_tool
)




load_dotenv(find_dotenv())
gemini_api_key = os.getenv("GEMINI_API_KEY")



# ======================== GEMINI KEYS STEP START ======================== 
provider = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" ,
    api_key=gemini_api_key,
)


model = OpenAIChatCompletionsModel(
    openai_client=provider,
    model="gemini-2.0-flash",
)


run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

# ======================== GEMINI KEYS STEP END ======================== 



# ================================ TOOLS ===============================

@function_tool
def get_advice(a: str) -> str:
    """This function will recieve a question and based on this i wil generate advice"""
    print(f"Agent call function with: {a}")
    return f"your question: {a}, My Advice: Never trust any one blindly... and follow health diet"




# ============================= SAME STEPS ====================================
agent = Agent(
    name="Career Mentor Agent",
    instructions="Provides help for academics questions if user ask for advice then use get_advice",
    tools=[get_advice]
)


result = Runner.run_sync(
    agent,
    input="i need advice for survive in job",
    run_config=run_config   # for GEMINI only
)

print(result.final_output)


