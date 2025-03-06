import os
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
# You can then use this template with LangChain's PromptTemplate:
from langchain.prompts import PromptTemplate

# Step1: Setup LLM
HF_TOKEN=os.environ.get("HF_TOKEN")

try:

    # Initialize HuggingFaceHub LLM (choose an appropriate model)
    llm = HuggingFaceHub(repo_id="chaoyi-wu/MedLLaMA_13B", model_kwargs={"temperature": 0.1, "max_length": 256})

    # Create PromptTemplate
    pregnant_medical_template = """Answer the medical question based on the pregnant woman's information below. If the question cannot be answered using the information provided, answer with "I cannot provide a diagnosis or medical advice based on the information given.".

    Context: You are a Pregnant woman with days that are filled with imagining the future, planning for the arrival of my little one, and trying to decipher the ever-changing signals my body sends. I crave connection, reassurance, and the shared understanding of other mothers. I am a blend of excitement and occasional anxieties,
             a woman carrying not just a child, but a universe of emotions within her.

    Medical Question: {medical_question}

    Answer: """

    prompt = PromptTemplate(template=pregnant_medical_template, input_variables=["medical_question"])

    # Example Usage:
    # pregnant_patient_info = "Age: 30, Gestational Age: 12 weeks, Symptoms: Mild nausea, fatigue, occasional heartburn. No known allergies. Due date: September 7th."
    # medical_question = "Are these symptoms normal for this stage of pregnancy?"


    # prompt = PromptTemplate(
    #     template=pregnant_medical_template,
    #     input_variables=["pregnant_patient_info", "medical_question"]
    # )

    # formatted_prompt = prompt.format(pregnant_patient_info=pregnant_patient_info, medical_question=medical_question)

    # print(formatted_prompt)

    # Create LLMChain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Run the chain
    medical_question =  "hi mama’s to be today I am 12 weeks and my Flo app says I’m about the size of a lemon but my belly " \
                        "feels so big lol my due date is September 7." \
                        "What is everybody else’s bellies looking like around 12 weeks? " \
                        "Is my baby growing fast or am I just bloated"
    
    result = llm_chain.run(medical_question)

    print("\nResult: ", result)

except ImportError:
    print("Error: Please install the required libraries: pip install langchain huggingface_hub transformers")
except ValueError as e:
    print(f"Error: {str(e)}. Please ensure your Hugging Face API token is set correctly and the model exists.")
except Exception as e:
    print(f"An unexpected error occurred: {str(e)}")

    
