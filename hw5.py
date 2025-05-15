import os
import webbrowser
import requests
import openai

# 步骤1：设置大模型API（这里以OpenAI为例）
API_KEY = "sk-0fbf3033883c45c39ee13b3939401132"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 步骤2：构建生成网页的提示词
PROMPT = """请生成一个完整的HTML网页，主题是：如何科学养猫。要求包含：
1. 响应式布局
2. 现代简约风格
3. 包含以下内容：
   - 导航栏（首页、简介、应用）
   - 主图区域
   - 三个特色卡片（原理、优势、挑战）
   - 页脚
4. 使用CSS渐变背景
5. 包含交互动画效果
请直接输出完整HTML代码，不要多余解释。"""

# 步骤3：调用大模型API
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "deepseek-reasoner",
    "messages": [
        {"role": "system", "content": "你是一个专业的全栈开发工程师，擅长生成现代网页设计"},
        {"role": "user", "content": PROMPT}
    ],
    "temperature": 0.7
}

response = requests.post(API_URL, headers=headers, json=data)
response_data = response.json()
print(response_data)

# 步骤4：提取生成的代码
generated_code = response_data['choices'][0]['message']['content']

# 步骤5：保存为HTML文件
output_path = "generated_website.html"
with open(output_path, "w", encoding="utf-8") as f:
    # 清理可能存在的代码块标记
    clean_code = generated_code.replace("```html", "").replace("```", "")
    f.write(clean_code)

# 步骤6：自动打开浏览器查看结果
webbrowser.open(f"file://{os.path.abspath(output_path)}")
