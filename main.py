# filepath: /home/firdaus/Documents/Projects/iNara-AI/livekit/main.py
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import google, cartesia, deepgram, noise_cancellation, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel
import asyncio
from dotenv import load_dotenv
import os
from livekit.agents import Agent, function_tool, RunContext
from rag.search import search_docs  # Pastikan fungsi ini mengembalikan string hasil pencarian

# Load environment variables from a .env file
load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="Kamu adalah Nara, AI Agent yang berperan sebagai staf TU di Universitas Kebangsaan Republik Indonesia.")

    @function_tool()
    async def retrieve_info(self, context: RunContext, query: str) -> str:
        """Mencari informasi dari basis data kampus berdasarkan pertanyaan pengguna."""
        results = search_docs(query)
        if results:
            return "\n".join(results)
        return "Maaf, saya tidak menemukan informasi yang relevan."



async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-live-001",
            voice="Leda",
            temperature=0.8,
            # instructions="You are a helpful assistant",
        ),
    )
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Kamu adalah Nara, AI Agent yang berperan sebagai staf TU di Universtas Kebangsaan Republik Indonesia."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))