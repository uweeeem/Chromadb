# Inspiration Search Engine

This project is an Inspiration Search Engine that allows users to find visually similar images based on a textual query. The application utilizes OpenAI's CLIP model for image and text feature extraction, and ChromaDB for managing image embeddings.

## Features

- **Search for Inspiration:** Users can enter a theme or idea to receive visually similar images.
- **Adjustable Results:** Users can specify the number of results to display and set a similarity threshold.
- **Dynamic Image Fetching:** The application fetches images from specified URLs and processes them in real-time.

## Technologies Used

- **Python**: Programming language for building the application.
- **Transformers**: For using the CLIP model for image and text embeddings.
- **ChromaDB**: For storing and querying image embeddings.
- **Gradio**: For creating the web interface.
- **PIL**: For image processing.
- **Requests**: For fetching images from the web.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository
