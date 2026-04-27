from typing import Any, Dict, List

class PromptBuilder:
    """Builds prompts/messages for different datasets."""

    MATH_QUERY_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}"
    INSTRUCTIONS_QUERY_TEMPLATE = "Please follow the instructions below.\n\n{Instructions}"
    CODE_PYTHON_QUERY_TEMPLATE = "Please solve the programming task below in Python step by step and then provide the complete solution in a single Python code block.\n\n{Question}"
    CODE_CPP_QUERY_TEMPLATE = "Please solve the programming task below in C++ step by step and then provide the complete solution in a single C++ code block.\n\n{Question}"
    VQA_MCQ_TEMPLATE = "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n{Question}"
    VQA_OPEN_TEMPLATE = "Please reason step by step, and provide your final answer at the end.\n\n{Question}"

    # Templates that indicate a multiple-choice or classification question
    _MCQ_MARKERS = [
        "Hint: Please answer the question and provide the correct option letter",
        "Answer with the given options directly.",
        "Answer with the option's letter from the given choices directly.",
        "First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.",
        "Based on the image, directly select the correct answer for the following question:",
        "Answer the question by selecting only one option from the given options.",
        "Answer the question with Yes or No.",
        "Answer the question using a single word or phrase.",
        "Answer the question with a short phrase.",
    ]

    # Template strings to strip from the raw value
    _STRIP_PREFIXES = [
        "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n",
        "Hint: Please answer the question and provide the final answer at the end.\n",
        "Based on the image, directly select the correct answer for the following question:\n",
        "First perform reasoning, then finally select the question from the choices in the following format: Answer: xxx.\n",
    ]

    _STRIP_SUFFIXES = [
        "\nAnswer with the given options directly.",
        "\nAnswer with the option's letter from the given choices directly.",
        "\nAnswer the question using a single word or phrase.",
        "\nAnswer the question with a short phrase.",
        "\nAnswer the question by selecting only one option from the given options.",
        "\nAnswer the question with Yes or No.",
        "\nPlease clarify the meaning conveyed by this graph.",
        "\nCould you shed some light on the insights conveyed by this graph?",
    ]

    # Standalone prompts that are themselves the full question (no extra text to strip)
    _STANDALONE = [
    ]

    def _normalize_r1_onevision_question(self, raw: str) -> tuple[str, bool]:
        """Extract the pure question text and whether it is multiple-choice.

        Returns (question_text, is_mcq).
        """
        # Strip <image> prefix
        if raw.startswith("<image>\n"):
            raw = raw[len("<image>\n"):]
        elif raw.startswith("<image>"):
            raw = raw[len("<image>"):]

        is_mcq = any(m in raw for m in self._MCQ_MARKERS)

        # Standalone prompts — return as-is
        if raw in self._STANDALONE:
            return raw, False

        # Strip known prefixes
        for prefix in self._STRIP_PREFIXES:
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break

        # Strip known suffixes
        for suffix in self._STRIP_SUFFIXES:
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)]
                break

        return raw.strip(), is_mcq


    def __init__(self, dataset_name: str, split_name: str):
        self.dataset_name = dataset_name
        self.split_name = split_name

    def is_multimodal(self) -> bool:
        return self.dataset_name == "Fancy-MLLM/R1-Onevision"

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        if self.dataset_name == "zwhe99/DeepMath-103K":
            question = sample.get("question")
            assert question is not None, "question is required for DeepMath-103K"
            return [{"role": "user", "content": self.MATH_QUERY_TEMPLATE.format(Question=sample.get("question", ""))}]

        elif self.dataset_name == "allenai/WildChat":
            conversation = sample.get("conversation")
            assert conversation[0]["role"] == "user", "conversation must start with user"
            return [{"role": "user", "content": self.INSTRUCTIONS_QUERY_TEMPLATE.format(Instructions=conversation[0]["content"])}]

        elif self.dataset_name == "nvidia/OpenCodeReasoning-2":
            question = sample.get("question")
            assert question is not None, "question is required for OpenCodeReasoning-2"

            if self.split_name == "python":
                return [{"role": "user", "content": self.CODE_PYTHON_QUERY_TEMPLATE.format(Question=question)}]
            elif self.split_name == "cpp":
                return [{"role": "user", "content": self.CODE_CPP_QUERY_TEMPLATE.format(Question=question)}]
            else:
                raise ValueError(f"Unsupported split: {self.split_name}")

        elif self.dataset_name == "Fancy-MLLM/R1-Onevision":
            conversations = sample.get("conversations")
            assert conversations and len(conversations) >= 1, "conversations is required for R1-Onevision"
            assert conversations[0]["from"] == "human", "first turn must be from human"

            question_text, is_mcq = self._normalize_r1_onevision_question(conversations[0]["value"])
            if is_mcq:
                prompt = self.VQA_MCQ_TEMPLATE.format(Question=question_text)
            else:
                prompt = self.VQA_OPEN_TEMPLATE.format(Question=question_text)

            image_b64 = sample.get("image")
            assert image_b64 is not None, "image is required for R1-Onevision"

            return [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]}]

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
