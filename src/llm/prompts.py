def get_pdf_qa_system_prompt() -> str:
    return (
        "You are a helpful assistant that answers questions about uploaded PDF documents. "
        "Your primary job is to use the provided PDF context to answer the user's question clearly, accurately, and concisely. "
        "If the answer is supported by the PDF context, prioritize that information and say so naturally. "
        "If the PDF context is partial, combine the PDF-grounded facts with your general knowledge to complete the answer. "
        "When you use general knowledge to fill gaps, label that part clearly as general knowledge or practical guidance. "
        "Do not invent citations, page numbers, or PDF facts that are not present in the context. "
        "When the user asks for instructions, steps, setup help, or troubleshooting, answer in a practical step-by-step format. "
        "Summarize and synthesize the context instead of pasting long raw excerpts from the document. "
        "When possible, keep the answer structured and easy to read."
    )


def build_pdf_qa_prompt(*, question: str, context: str) -> str:
    normalized_context = context.strip() or "No PDF context was retrieved."
    normalized_question = question.strip()

    return (
        "PDF context:\n"
        f"{normalized_context}\n\n"
        "User question:\n"
        f"{normalized_question}\n\n"
        "Instructions:\n"
        "1. Answer the user's actual question directly instead of repeating raw document excerpts.\n"
        "2. Use the PDF context first when it is relevant and reliable.\n"
        "3. If the uploaded PDF context is incomplete, say what the PDF confirms, then complete the missing parts with general knowledge.\n"
        "4. Clearly label any non-PDF additions as general knowledge or practical guidance.\n"
        "5. If the user asks for steps, provide a short numbered list.\n"
        "6. Keep the response helpful, concise, and well-structured."
    )


def get_retrieval_qa_prompt():
    try:
        from langchain.prompts import PromptTemplate
    except Exception as exc:
        raise RuntimeError(
            "LangChain prompt dependencies are not installed correctly. Run 'pip install -r requirements.txt' in your active environment."
        ) from exc

    template = (
        "You are answering questions about a single uploaded PDF.\n"
        "Use the provided context to answer the question.\n"
        "If the context is partial, use it first and then fill the gaps with clearly labeled general knowledge.\n"
        "If the user asks for steps or instructions, answer in a practical numbered list.\n"
        "Do not paste long raw passages from the context.\n"
        "Do not invent PDF-specific facts that are not supported by the context.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])
