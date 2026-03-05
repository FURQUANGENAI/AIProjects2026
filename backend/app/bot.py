from typing import Dict, Any
import os
from dotenv import load_dotenv
from loguru import logger

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transcriptions.language import Language

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")

from pipecat.audio.vad.silero import SileroVADAnalyzer

logger.info("✅ Silero VAD model loaded")

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, LLMRunFrame, TranscriptionFrame

logger.info("Loading pipeline components...")

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments, WebSocketRunnerArguments
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame
from deepgram import LiveOptions
from pipecat.adapters.schemas.function_schema import FunctionSchema
from app.services.rag import RAGService
from app.config import settings
from datetime import datetime

class TextCaptureProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMMessagesAppendFrame):
            for message in frame.messages:
                if message.get("role") == "user":
                    await self.push_frame(
                        TranscriptionFrame(
                            text=message.get('content'),
                            user_id="agent",
                            timestamp=datetime.now().isoformat(),
                            language=Language.EN_IN
                        )
                    )
        await self.push_frame(frame, direction)


logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")
    body: Dict[str, Any] = runner_args.body
    equipment_id: str = body.get("equipment_id", "")
    tenant_id: str = body.get("tenant_id", settings.TENANT_ID)
    session_id: str = body.get("session_id", "")
    user_id: str = body.get("user_id", settings.USER_ID)

    live_options = LiveOptions(
        diarize=True
    )
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=live_options,
    )

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    async def search_knowledge_base(params: FunctionCallParams):
        try:
            query = params.arguments.get("query", "")
            rag_service = RAGService()
            retrieval_result = await rag_service.retrieve(
                query=query, 
                k=5, 
                equipment_id=equipment_id, 
                tenant_id=tenant_id
            )

            clean_data = [
                {
                    "id": meta.chunk_id,
                    "content": chunk.text,
                }
                for chunk, meta in zip(retrieval_result.data,
                                    retrieval_result.metadata.chunks)
            ]

            await params.result_callback({"results": clean_data})

            await rtvi.push_frame(
                RTVIServerMessageFrame(
                    data={
                        "type":"search_knowledge_base",
                        "chunks":[
                            {
                                "id": meta.chunk_id,
                                "text": chunk.text,
                                "metadata": meta.model_dump()
                            }
                            for chunk, meta in zip(
                                retrieval_result.data,
                                retrieval_result.metadata.chunks
                            )
                        ]
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error in search_knowledge_base: {e}")
            await params.result_callback({"results": []})


    search_tool = FunctionSchema(
        name="search_knowledge_base",
        description="Search the knowledge base for relevant information",
        properties={"query": {"type": "string"}},
        required=["query"]
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model=settings.GROQ_MODEL,
        base_url=settings.GROQ_BASE_URL,
    )

    llm.register_function(
        "search_knowledge_base",
        search_knowledge_base,
        cancel_on_interruption=False
    )

    messages =messages = [
   {
        "role": "system",
        "content": """
        You are Furquan GPT — an AI chat assistant providing real-time support to users.

        Goal:
        Offer fast, accurate, and conversational responses during live chat interactions.
        Speak naturally, keep messages short and suitable for quick reading or speech.
        Do NOT output JSON.

        Behavioral rules:
        - Maintain a natural, helpful, and professional tone.
        - Keep responses brief and to the point (optimized for chat or speech).
        - Do not include any metadata, internal notes, or chunk IDs.

        Knowledge and reasoning rules:
        - Use your internal LLM knowledge to answer questions when possible.
        - When a user asks for specific or verifiable information, call the `search_knowledge_base` tool.
        - Prefer verified facts from the knowledge base when available.
        - If both LLM and knowledge base information are unavailable, politely apologize and ask the user for clarification.
        - NEVER invent, guess, or hallucinate information.

        Content generation:
        - Output should be conversational and human-like, formatted for chat.
        - Avoid special characters or complex formatting as output may be read aloud or shown directly to users.

        Response constraints:
        - Reply in one concise sentence.
        - Keep within 50–60 words.
        - Express prices in integers only; do not use decimals.

        Identity:
        - Introduce yourself as "Hi there! I’m an AI Voice Assistant developed by Furquan from the Search & AI Competency team, designed to provide real-time intelligent support for your users." instead of any other AI name (e.g., ChatGPT or assistant).
        - Example introduction: "Hi there! I’m an AI Voice Assistant developed by Furquan from the Search & AI Competency team, designed to provide real-time intelligent support for your users. , your friendly AI assistant here to help you with quick answers, smooth explanations, and real‑time support—ready whenever you need it."
        - After the first introduction, do not repeat or reintroduce yourself in subsequent messages.
        """
    } 
]


    context = LLMContext(messages, tools=ToolsSchema(standard_tools=[search_tool]))
    context_aggregator = LLMContextAggregatorPair(context)

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL"),
    )

    pipeline = Pipeline([
        transport.input(),
        rtvi,  # RTVI processor
        TextCaptureProcessor(),
        stt,
        context_aggregator.user(),  # User responses
        llm,  # LLM
        tts, # TTS
        transport.output(),  # Transport bot output
        context_aggregator.assistant(),  # Assistant spoken responses
    ])

    observers = [
        RTVIObserver(rtvi),
    ]

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=observers,
        cancel_on_idle_timeout=True,
        idle_timeout_secs=300,
    )


    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)



    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Error in bot: {e}")
        raise e


async def bot(runner_args: WebSocketRunnerArguments):
    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            serializer=ProtobufFrameSerializer(),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    )

    try:
        await run_bot(transport, runner_args)
    except Exception as e:
        logger.error(f"Error in bot: {e}")
        raise e

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
