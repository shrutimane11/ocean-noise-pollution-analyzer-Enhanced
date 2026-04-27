import sys
sys.path.append(".")

from src.leaderboard import initialize_leaderboard, add_entry

# Create leaderboard if not exists
initialize_leaderboard()

# Add sample models
add_entry("CNN_Model", 0.91, 0.89, 0.88, 0.90)
add_entry("LSTM_Model", 0.88, 0.86, 0.85, 0.87)

print("Leaderboard updated successfully!")