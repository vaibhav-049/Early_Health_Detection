<<<<<<< HEAD
# Early_Health_Detection
=======
# EarlyHealthDetector

EarlyHealthDetector is a Flask-based web application for early health risk detection using machine learning. It provides a simple web interface for users to input health data and receive predictions.

## Features

- User-friendly web interface
- Machine learning model for health risk prediction
- Data visualization
- SQLite/PostgreSQL database support

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

1. Clone the repository:
   ```powershell
   git clone https://github.com/your-username/EarlyHealthDetector.git
   cd EarlyHealthDetector
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Set the session secret and run the app:
   ```powershell
   $env:SESSION_SECRET="your_secret_key"
   python main.py
   ```
4. Open your browser and go to http://localhost:5000

## Project Structure

- `app.py` - Main Flask app setup
- `main.py` - Entry point
- `models.py` - Database models
- `routes.py` - App routes
- `utils.py` - Utility functions
- `static/` - Static files (CSS, JS)
- `templates/` - HTML templates
- `data/` - Sample data

## License

This project is open source and available under the MIT License.
>>>>>>> fdbfeea (Push all project files to GitHub)
