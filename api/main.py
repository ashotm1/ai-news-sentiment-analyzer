from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from scraper.stocktitan import scrape

app = FastAPI()
SECRET = "1234"


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>News Sentiment Analyzer</h2>

            <input id="count" type="number" min="1" max="30" value="10"/>

            <button onclick="run()">Analyze</button>

            <pre id="output"></pre>

            <script>
            async function run() {
                const count = document.getElementById("count").value;

                const res = await fetch("/run", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "x-key": "1234"
                    },
                    body: JSON.stringify({limit: parseInt(count)})
                });

                const data = await res.json();
                document.getElementById("output").innerText =
                    JSON.stringify(data, null, 2);
            }
            </script>
        </body>
    </html>
    """


@app.post("/run")
async def run(request: Request):
    if request.headers.get("x-key") != SECRET:
        return {"error": "unauthorized"}

    body = await request.json()
    limit = body.get("limit", 10)

    return {
        "articles": scrape(limit)
    }