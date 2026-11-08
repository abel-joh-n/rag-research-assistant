from src.gemini import generate_answer


class RAGPipeline:

    def __init__(self, retriever):
        self.retriever = retriever

    def ask(self, query, k=3):

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query,
            k=k
        )

        # Extract the text from retrieved chunks
        context = "\n\n".join(
            result["text"]
            for result in results
        )

        # Generate answer using Gemini
        answer = generate_answer(
            query,
            context
        )

        return answer, results