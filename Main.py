# Importing the needed modules 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
import json, os
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Loading API key from a dotenv file.
load_dotenv(dotenv_path='.env')

# --- IMPORTANT: Explicitly unset OPENAI_API_KEY if it exists ---
# This ensures no accidental fallback to OpenAI if we intend to use only Groq
if "OPENAI_API_KEY" in os.environ:
    print("DEBUG: Unsetting OPENAI_API_KEY to ensure Groq is used.")
    del os.environ["OPENAI_API_KEY"]


# --- DIAGNOSTIC STEP: VERIFY GROQ API KEY LOADING ---
groq_key_check = os.getenv("GROQ_API_KEY")
if groq_key_check:
    print(f"DEBUG: GROQ_API_KEY successfully loaded. Key starts with: {groq_key_check[:5]}...")
else:
    print("ERROR: GROQ_API_KEY not found after loading .env. Please check:")
    print("  1. Is your .env file named exactly '.env'?")
    print("  2. Is your .env file in the same directory as this script?")
    print("  3. Does your .env file contain GROQ_API_KEY=\"gsk_...\"?")
    print("Exiting as API key is essential.")
    exit(1) # Exit the script if the key isn't loaded
# ------------------------------------------------

# read the medical report
with open(r"Medical Reports\Medical Rerort - Michael Johnson - Panic Attack Disorder.txt", "r") as file:
    medical_report = file.read()


agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}

# Function to run each agent and get their response
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

# Run the agents concurrently and collect responses
responses = {}
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_response, name, agent): name for name, agent in agents.items()}
    
    for future in as_completed(futures):
        agent_name, response = future.result()
        if response is None:
            print(f"WARNING: {agent_name} agent returned None. Skipping its report.")
            responses[agent_name] = f"Error: {agent_name} report could not be generated." # Provide a fallback string
        else:
            responses[agent_name] = response

required_agents = ["Cardiologist", "Psychologist", "Pulmonologist"]
missing_reports = [agent for agent in required_agents if agent not in responses or responses[agent] is None]

if missing_reports:
    print(f"ERROR: Missing reports for the following agents: {', '.join(missing_reports)}. Cannot proceed with MultidisciplinaryTeam.")
    final_diagnosis = "Failed to generate final diagnosis due to missing individual agent reports."
else:
    team_agent = MultidisciplinaryTeam(
        cardiologist_report=responses["Cardiologist"],
        psychologist_report=responses["Psychologist"],
        pulmonologist_report=responses["Pulmonologist"]
    )

    final_diagnosis = team_agent.run() 

if final_diagnosis is None:
    print("ERROR: MultidisciplinaryTeam agent failed to generate a final diagnosis.")
    final_diagnosis_text = "### Final Diagnosis:\n\nFailed to generate a comprehensive diagnosis due to an internal error."
else:
    final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis

txt_output_path = "results/final_diagnosis.txt"

os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

with open(txt_output_path, "w") as txt_file:
    txt_file.write(final_diagnosis_text)

print(f"Final diagnosis has been saved to {txt_output_path}")
