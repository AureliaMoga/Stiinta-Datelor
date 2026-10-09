from pathlib import Path
from datetime import datetime

# Target directory
output_dir = Path("/sentiment_ro")


# Start and end dates
start = datetime(2019, 1, 1)
end = datetime(2026, 5, 1)

current = start

while current <= end:
    filename = current.strftime("%Y-%m.json")
    file_path = output_dir / filename

    # Create empty JSON file if it does not already exist
    if not file_path.exists():
        file_path.write_text("{}")

    # Move to next month
    if current.month == 12:
        current = current.replace(year=current.year + 1, month=1)
    else:
        current = current.replace(month=current.month + 1)

print("Empty JSON files created successfully.")