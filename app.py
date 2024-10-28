# Import necessary libraries
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
import requests
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# Setup ChromaDB
client = chromadb.Client()
collection = client.create_collection("inspiration_collection")

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# List of alternative image URLs
image_urls = [ 
    "https://artfasad.com/wp-content/uploads/2024/01/l-shaped-modern-kitchen-design-22.jpg",  # Modern Kitchen
    "https://media.designcafe.com/wp-content/uploads/2020/05/09150825/blue-and-white-modular-kitchen-design.jpg",  # Interior Design
    "https://marieclaire.ng/wp-content/uploads/2023/10/TFSF-FI.jpg",  # Fashion Ideas
    "https://pragueviews.com/wp-content/uploads/2024/04/Mural-Karlin-1-DSC_7376-1.jpg",  # Street Art
    "https://fancyhouse-design.com/wp-content/uploads/2024/01/A-beautiful-great-room-with-modern-design-elements.jpg"
]

# Headers for requests to mimic a browser
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Accept': 'image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# Fetch and process images
images = []
for url in image_urls:
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Check if the response content type is an image
        if 'image' in response.headers.get('Content-Type', ''):
            img = Image.open(BytesIO(response.content)).convert("RGB")  # Ensure image is in RGB format
            images.append(img)
        else:
            print(f"Warning: URL does not point to an image: {url}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {url}: {e}")
    except Exception as e:
        print(f"Failed to process image from {url}: {e}")

# Check if images were fetched successfully
if images:
    # Prepare images for the model
    inputs = processor(images=images, return_tensors="pt", padding=True)

    # Move inputs to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get image features
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs).cpu().numpy()  # Move to CPU for numpy
    image_embeddings = [embedding.tolist() for embedding in image_embeddings]

    # Add embeddings to ChromaDB
    collection.add(
        embeddings=image_embeddings,
        metadatas=[{"image": url} for url in image_urls],
        ids=[str(i) for i in range(len(image_urls))]
    )
else:
    print("No valid images were fetched. Please check the URLs.")

# Function to calculate similarity score
def calculate_similarity(image_embedding, query_embedding):
    return cosine_similarity([image_embedding], [query_embedding])[0][0]

# Function to fetch images based on a theme query
def search_image(query, result_count=5, similarity_threshold=0.2):
    # Process the text query
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).cpu().numpy().tolist()

    # Query the collection
    results = collection.query(query_embeddings=query_embedding, n_results=result_count)
    top_results = []
    
    for result in results['metadatas'][0]:
        matched_image_url = result.get('image')
        matched_image_index = result.get('id')

        # Check if 'id' exists in result
        if matched_image_index is not None:
            matched_image_embedding = image_embeddings[int(matched_image_index)]
            similarity = calculate_similarity(matched_image_embedding, query_embedding[0])

            if similarity >= similarity_threshold:
                response = requests.get(matched_image_url)
                result_image = Image.open(BytesIO(response.content))
                top_results.append((result_image, f"Similarity score: {similarity:.4f}"))

    if not top_results:
        return None, "No matches found. Try another search term."
    
    return top_results[0][0], f"Suggested images for '{query}'"

# Gradio Interface
with gr.Blocks() as gr_interface:
    gr.Markdown("# Inspiration Search Engine")
    custom_query = gr.Textbox(placeholder="Type an idea or theme (e.g., modern living room)", label="Get inspiration for:")
    result_count = gr.Slider(1, 10, step=1, value=5, label="Number of Results")
    similarity_threshold = gr.Slider(0.0, 1.0, step=0.1, value=0.2, label="Similarity Threshold")
    
    submit_button = gr.Button("Get Inspiration")
    image_output = gr.Image(type="pil", label="Inspiration")
    description_output = gr.Textbox(label="Description")

    submit_button.click(fn=search_image, inputs=[custom_query, result_count, similarity_threshold], outputs=[image_output, description_output])

gr_interface.launch(share=True)
