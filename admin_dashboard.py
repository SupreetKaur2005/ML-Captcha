import os
import logging
from flask import Flask, jsonify, render_template
from database import CaptchaDatabase
import matplotlib.pyplot as plt
import io
import base64

# Configure logging
logging.basicConfig(
    filename='logs/logfile.log',  # Log file path
    level=logging.DEBUG,          # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__, template_folder='templates')
db = CaptchaDatabase()

@app.route('/sessions', methods=['GET'])
def get_sessions():
    """Fetch user sessions from the database."""
    logging.info("Fetching user sessions from the database.")
    try:
        sessions = db.get_all_sessions()  # Assuming this method exists
        logging.debug(f"Fetched sessions: {sessions}")
        return render_template('admin_sessions.html', sessions=sessions)
    except Exception as e:
        logging.error(f"Error fetching sessions: {e}")
        return jsonify({"error": "Could not fetch sessions"}), 500

@app.route('/visualization', methods=['GET'])
def visualize_data():
    """Generate a simple visualization of user behavior."""
    logging.info("Generating user behavior visualization.")
    try:
        # Example data
        labels = ['Bots', 'Humans']
        sizes = [db.get_bot_count(), db.get_human_count()]  # Assuming these methods exist

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close(fig)

        logging.debug("Visualization generated successfully.")
        return render_template('visualization.html', image=image_base64)
    except Exception as e:
        logging.error(f"Error generating visualization: {e}")
        return jsonify({"error": "Could not generate visualization"}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    """Fetch and display logs."""
    log_file_path = 'logs/logfile.log'
    log_dir = os.path.dirname(log_file_path)

    try:
        # Check if the log directory exists, if not, create it
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logging.info("Log directory created.")

        # Check if the log file exists, if not, create it
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w') as f:
                f.write("Log file created.\n")
            logging.info("Log file created.")

        # Read the log file
        with open(log_file_path, 'r') as f:
            logs = f.readlines()

        logging.debug("Logs fetched successfully.")
        return render_template('logs.html', logs=logs)
    except Exception as e:
        logging.error(f"Error fetching logs: {e}")
        return jsonify({"error": "Could not fetch logs"}), 500

if __name__ == '__main__':
    logging.info("Starting admin_dashboard application.")
    app.run(debug=True)