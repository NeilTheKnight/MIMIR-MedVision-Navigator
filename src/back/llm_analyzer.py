import os
import base64
from openai import OpenAI


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_qwen(image_path: str, question: str = None) -> str:
    """调用 DashScope 兼容接口的 Qwen-VL 模型分析图片。

    环境变量：
    - DASHSCOPE_API_KEY：API Key（必需）
    - QWEN_VL_MODEL：模型名称，默认"qwen-vl-plus"，可设置为"qwen3-vl-plus"。
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    if not client.api_key:
        raise RuntimeError("未设置环境变量 DASHSCOPE_API_KEY")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    base64_image = encode_image(image_path)

    if question is None:
        question = (
            "说明从图片可见的肿瘤阶段（基于影像学形态、大小的初步判断），以及该肿瘤可能造成的后果（基于肿瘤位置的潜在影响）\n"
            "判定依据\n"
            "1肿瘤位置：明确肿瘤在脑部的具体区域（如鞍区等）\n"
            "2肿瘤形态：描述肿瘤的影像学形态特征（如团块状等）\n"
            "3肿瘤与周围组织关系：说明肿瘤与周围脑部结构的位置关联\n"
            "4阶段判断依据：基于上述位置、形态、大小等信息，说明判断肿瘤阶段的理由\n"
            "5后果分析依据：基于肿瘤位置和可能的生长趋势，说明可能造成的神经功能影响、压迫症状等后果的推理逻辑"
        )

    model_name = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
    )

    return completion.choices[0].message.content