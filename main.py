from inference.predict_score import GetScore

if __name__ == "__main__":
    user_file = "input xml here"
    score = GetScore(user_file)
    print(f"Predicted Sleep Score: {score['SleepScore']}")