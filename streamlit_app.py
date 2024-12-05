import streamlit as st
from helpers.rag import perform_rag
from helpers.analysis import retrieve_articles

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Groq's Llama 3.1"
    
retrieve_articles()

# Function to select AI model
def select_model():
    model_options = ["Groq's Llama 3.1", "Google Gemini"]
    selected_model = st.selectbox(
        "Select AI model:",
        options=model_options,
        index=model_options.index(st.session_state.selected_model)
    )

    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model

def main():
    st.title("Financial Analysis Automation")
    # Display previous messages
    if st.session_state.messages:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    image = None
    if not st.session_state.selected_model == "Groq's Llama 3.1":
        image = st.file_uploader("Upload an image for multi-modal input: (Optional)", type=["png", "jpg", "jpeg"])

    # Display preview of image uploaded
    if image:
        st.image(image, caption="Uploaded Image")
        st.write(f"File type: {image.type}, Name: {image.name}, Size: {image.size}")
        st.success("Image uploaded successfully!")
        
    prompt = st.chat_input("Chat...")
        
    if prompt:
        if st.session_state.selected_model == "Groq's Llama 3.1":
            llm_response = perform_rag(prompt, "Groq's Llama 3.1")
        else:
            llm_response = perform_rag(prompt, "Google Gemini", image=image)

        st.session_state.messages.append({"role": "assistant", "content": llm_response})
        with st.chat_message("assistant"):
            st.markdown(llm_response)
    elif image and not prompt:
        st.warning("Please provide a text prompt along with the image.")

# Sidebar for codebase and model selection
with st.sidebar:
    st.header("AI Model Selection")
    select_model()

# Run the main function
if __name__ == "__main__":
    main()

