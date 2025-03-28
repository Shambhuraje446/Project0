from fastapi import FastAPI
from IPLBettingTool import IPLBettingTool
from pydantic import BaseModel

app = FastAPI()

betting_tool = IPLBettingTool(data_path='matches.csv')

class MatchPredictionRequest(BaseModel):
    team1: str
    team2: str
    venue: str
    toss_winner: str
    toss_decision: str

@app.get("/")
async def root():
    return {"message": "Welcome to IPL Betting Insights API!"}

@app.post("/predict-match")
async def predict_match(request: MatchPredictionRequest):
    prediction = betting_tool.predict_match(
        request.team1,
        request.team2,
        request.venue,
        request.toss_winner,
        request.toss_decision
    )
    return prediction

@app.get("/available-teams")
async def available_teams():
    return betting_tool.get_available_teams()

@app.get("/available-venues")
async def available_venues():
    return betting_tool.get_available_venues()

