import os
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

# Set your Hugging Face API token as an environment variable (replace with your actual token)
# Alternatively set it in your terminal as: export HUGGINGFACEHUB_API_TOKEN="your_token_here"
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token_here"

try:
    # Initialize HuggingFaceHub LLM (choose an appropriate model)
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.1, "max_length": 256})

    # Create PromptTemplate
    template = """Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know".
    Context: Libraries are places full of books.
    Question: {question}
    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Create LLMChain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Run the chain
    question = "Which libraries and model providers offer LLMs?"
    result = llm_chain.run(question)

    print(result)

except ImportError:
    print("Error: Please install the required libraries: pip install langchain huggingface_hub transformers")
except ValueError as e:
    print(f"Error: {e}. Please ensure your Hugging Face API token is set correctly and the model exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")