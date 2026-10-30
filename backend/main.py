from fastapi import FastAPI,Request,Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pypinyin import lazy_pinyin, Style
from snownlp import SnowNLP
from zhipuai import ZhipuAI
import os, json, re
from dotenv import load_dotenv
from storage import init_db,save_record,get_history
from datetime import datetime, timezone
import uuid
import os
from dotenv import load_dotenv

load_dotenv()                        # ← 读同目录下的 .env

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS").split(",")

init_db()



load_dotenv()  # 读 backend/.env
client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))

GLM_PROMPT = """你是中文情感分析器。分析用户文本的情绪。
只输出一个 JSON 对象，不要任何多余文字，格式严格如下：
{"score": 0到1之间的小数, "label": "偏积极/中性/偏消极", "reason": "一句简短理由"}
score 越接近 1 越积极，越接近 0 越消极。"""


# 中间件，解决了cors跨域问题
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=True,
)

def get_session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get("session_id")      # 先看有没有纸条
    if not sid:                                  # 第一次来，没有——发一张
        sid = uuid.uuid4().hex                    # 一串随机、不重复的 id
        response.set_cookie(
            "session_id", sid,
            httponly=True, samesite="lax",
            max_age=60 * 60 * 24 * 30,            # 记 30 天
        )
    return sid


profile = {
        "heroTitle": "关于我(来自后端)",
            "heroSubtitle": "项目，创意，灵感，心得，我的作品",
            "featuredWork": {
            "kicker": "作品",
            "title": "文字实验室",
            "copy": "拼音和情绪，挖掘中文里的细节",
            "linkLabel": "打开作品",
        },
        "identity": {
            "motto": "已识乾坤大，尤怜草木青",
            "learning": "零到全栈",
  },
}
class AnalyzeRequest(BaseModel):
    text: str


@app.get("/api/profile")
def get_profile():
    return profile

def score_label(score):
    if score >= 0.6:
        return "偏积极"
    elif score <= 0.4:
        return "偏消极"
    else:
        return "中性"

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, request: Request, response: Response):
    sid = get_session_id(request, response)
    text = req.text
    try:
        # 优先用 GLM 算情绪
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": GLM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        match = re.search(r"\{.*\}", content, re.S)
        data = json.loads(match.group(0))
        score = float(data["score"])
        label = score_label(score)
    except Exception:
        # GLM 调用失败时，退回 SnowNLP，界面不崩
        score = round(SnowNLP(text).sentiments, 2)
        label = score_label(score)

    result = {
        "text": req.text,
        "score": round(score, 2),
        "label": label,
        "pinyin": " ".join(lazy_pinyin(text, style=Style.TONE)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # ← 新增
    }
    save_record(sid,result)                                                          # ← 存档到文件
    return result

@app.get("/api/history")
def history(request: Request, response: Response, limit: int = 10):
    sid = get_session_id(request, response)
    return get_history(sid, limit) 