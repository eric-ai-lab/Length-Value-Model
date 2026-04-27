from enum import Enum, auto


class ModelType(Enum):
    GPT_4o_mini = auto()
    GPT_4o = auto()
    GPT_41_mini = auto()
    GPT_41 = auto()
    GPT_54 = auto()
    GPT_54_thinking = auto()
    o1_mini = auto()
    o4_mini = auto()
    o3_mini = auto()
    Claude_35_sonnet = auto()
    Claude_37_sonnet = auto()
    Claude_37_sonnet_thinking = auto()
    Claude_46_sonnet = auto()
    Claude_46_sonnet_thinking = auto()
    Claude_46_opus = auto()
    Claude_46_opus_thinking = auto()
    Gemini_2_flash = auto()
    Gemini_2_flash_thinking = auto()
    Gemini_25_flash = auto()
    Gemini_25_flash_lite = auto()
    Gemini_25_pro = auto()
    Gemini_3_flash = auto()
    Gemini_3_pro = auto()
    Gemini_31_flash_lite = auto()
    Gemini_31_pro = auto()
    Claude_4_sonnet = auto()
    Claude_4_opus = auto()
    Doubao_15_pro = auto()
    Doubao_15_thinking_pro = auto()
    Deepseek_V3 = auto()
    Deepseek_R1 = auto()
    Llama31_8B = auto()
    Llama31_70B = auto()
    Qwen3_235B = auto()
    Qwen3_235B_Thinking = auto()
    Qwen3_235B_Instruct_2507 = auto()
    Qwen3_32B = auto()
    Qwen3_32B_Thinking = auto()
    Qwen25_3B = auto()
    Qwen25_7B = auto()
    Qwen25_72B = auto()
    GLM4_9B = auto()
    Mistral_7B = auto()
    LongWriter_GLM4_9B = auto()
    LongWriter_Llama31_8B = auto()
    Suri = auto()
    SGLang_Local = auto()
    SGLang_Local_No_Length_Control = auto()
    SGLang_Local_1 = auto()
    @staticmethod
    def get_api_type(model_type):
        api_type_dict = {
            ModelType.GPT_4o_mini: "gpt-4o-mini",
            ModelType.GPT_4o: "gpt-4o",
            ModelType.GPT_41_mini: "gpt-4.1-mini",
            ModelType.GPT_41: "gpt-4.1",
            ModelType.GPT_54: "gpt-5.4",
            ModelType.GPT_54_thinking: "gpt-5.4-thinking",
            ModelType.o1_mini: "o1-mini",
            ModelType.o4_mini: "o4-mini",
            ModelType.o3_mini: "o3-mini",
            ModelType.Claude_35_sonnet: "claude-3.5-sonnet",
            ModelType.Claude_37_sonnet: "claude-3.7-sonnet",
            ModelType.Claude_37_sonnet_thinking: "claude-3.7-sonnet-thinking",
            ModelType.Claude_46_sonnet: "claude-sonnet-4-6",
            ModelType.Claude_46_sonnet_thinking: "claude-sonnet-4-6-thinking",
            ModelType.Claude_46_opus: "claude-opus-4-6",
            ModelType.Claude_46_opus_thinking: "claude-opus-4-6-thinking",
            ModelType.Gemini_2_flash: "gemini-2.0-flash",
            ModelType.Gemini_2_flash_thinking: "gemini-2.0-flash-thinking",
            ModelType.Gemini_25_flash: "gemini-2.5-flash",
            ModelType.Gemini_25_flash_lite: "gemini-2.5-flash-lite",
            ModelType.Gemini_25_pro: "gemini-2.5-pro",
            ModelType.Gemini_3_flash: "gemini-3-flash-preview",
            ModelType.Gemini_3_pro: "gemini-3-pro-preview",
            ModelType.Gemini_31_flash_lite: "gemini-3.1-flash-lite-preview",
            ModelType.Gemini_31_pro: "gemini-3.1-pro-preview",
            ModelType.Claude_4_sonnet: "claude-sonnet-4",
            ModelType.Claude_4_opus: "claude-opus-4",
            ModelType.Doubao_15_pro: "doubao-1.5-pro",
            ModelType.Doubao_15_thinking_pro: "doubao-1.5-thinking-pro",
            ModelType.Deepseek_V3: "deepseek-v3",
            ModelType.Deepseek_R1: "deepseek-r1",
            ModelType.Llama31_8B: "Llama-3.1-8B-Instruct",
            ModelType.Llama31_70B: "Llama-3.1-70B-Instruct",
            ModelType.Qwen3_235B: "Qwen3-235B-A22B",
            ModelType.Qwen3_235B_Thinking: "Qwen3-235B-A22B-Thinking",
            ModelType.Qwen3_235B_Instruct_2507: "Qwen3-235B-A22B-Instruct-2507",
            ModelType.Qwen3_32B: "Qwen3-32B",
            ModelType.Qwen3_32B_Thinking: "Qwen3-32B-Thinking",
            ModelType.Qwen25_3B: "Qwen2.5-3B-Instruct",
            ModelType.Qwen25_7B: "Qwen2.5-7B-Instruct",
            ModelType.Qwen25_72B: "Qwen2.5-72B-Instruct",
            ModelType.GLM4_9B: "glm-4-9b",
            ModelType.Mistral_7B: "Mistral-7B-Instruct",
            ModelType.LongWriter_GLM4_9B: "LongWriter-glm4-9b",
            ModelType.LongWriter_Llama31_8B: "LongWriter-llama3.1-8b",
            ModelType.Suri: "suri-i-orpo",
            ModelType.SGLang_Local: "sglang-local",
            ModelType.SGLang_Local_1: "sglang-local-1",
            ModelType.SGLang_Local_No_Length_Control: "sglang-local-no-length-control",
        }
        return api_type_dict[model_type]
