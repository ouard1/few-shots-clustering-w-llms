# Replia Customer Support Chat

A modern customer support chat application powered by AI that provides quick, accurate answers to customer queries.

## Project Structure

- `index.html` - Main HTML file for the frontend
- `styles.css` - CSS styling for the frontend
- `script.js` - JavaScript for the frontend functionality
- `api/` - Directory containing the backend API
  - `app.py` - Flask application for the backend
  - `requirements.txt` - Python dependencies for the backend
  - `chroma_db/` - Database files for the vector database
  - Other model files and data

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd customer-support-page/api
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Run the backend server:
   ```
   python app.py
   ```
   The backend will run on `http://localhost:5000`

### Frontend Setup

Since the frontend is static HTML/CSS/JS, you can simply open the HTML file in a browser or use a simple HTTP server:

1. Using Python's built-in HTTP server:
   ```
   cd customer-support-page
   python -m http.server
   ```
   Then access the app at `http://localhost:8000`

2. Or simply open the `index.html` file in your browser.

## API Endpoints

- `/query` - POST endpoint for querying the AI assistant
  - Request body: `{ "query": "Your question here" }`
  - Response: `{ "response": "AI response here" }`

## Notes

- The backend uses the Mistral AI model for generating responses
- The chat interface is connected to the backend API automatically
- API keys in the code should be kept secret in a production environment

## Features

- Clean, modern UI 
- Interactive chat interface with realistic typing indicators
- Attractive hero section with gradient text effects
- Mobile-friendly navigation

## Technologies Used

- HTML5
- CSS3 (with modern features like CSS variables, flexbox, and grid)
- Vanilla JavaScript (no frameworks)
- Font Awesome icons



## License

This project is available for personal and commercial use. 