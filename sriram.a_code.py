import os
import zipfile
import pandas as pd
import numpy as np
import cv2  # OpenCV library for image and video processing
import torch  # Main PyTorch library
import torch.nn as nn  # PyTorch module for neural network components
import torch.optim as optim  # PyTorch module for optimization algorithms
from torch.utils.data import Dataset, DataLoader  # PyTorch utilities for data handling
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Huggingface transformers for NLP models and tokenization
from sklearn.metrics import f1_score, precision_score, recall_score  # Metrics for evaluation

# Define paths and filenames
zip_file_path = '/Users/aswinlakshmanan/Desktop/sample.zip'  # Path to the zip file containing video files
extract_dir = '/Users/aswinlakshmanan/Desktop/extracted_videos/'  # Directory to extract video files to
textual_data_file = '/Users/aswinlakshmanan/Desktop/textual_data.csv'  # Path to the CSV file containing textual data
ground_truth_file = '/Users/aswinlakshmanan/Desktop/ground_truth.csv'  # Path to the CSV file containing ground truth data

# Function to extract .mp4 files from zip archive
def extract_video_files(zip_file, extract_to):
    """Extract .mp4 files from a zip archive."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:  # Open the zip file
        zip_ref.extractall(extract_to)  # Extract all files to the specified directory

# Load textual data
df_text = pd.read_csv(textual_data_file)  # Read the CSV file containing textual data into a DataFrame

# Load ground truth data
df_ground_truth = pd.read_csv(ground_truth_file)  # Read the CSV file containing ground truth data into a DataFrame

# Extract video files from zip archive
extract_video_files(zip_file_path, extract_dir)  # Call the function to extract video files

# List all .mp4 files in the extraction directory
video_files = []  # Initialize an empty list to store video file paths
for file in os.listdir(extract_dir):  # Loop through all files in the extraction directory
    if file.endswith('.mp4'):  # Check if the file has an .mp4 extension
        video_files.append(os.path.join(extract_dir, file))  # Add the full file path to the list

# Device selection (use GPU if available, otherwise use CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select the device for computation

# Define custom dataset class for combining text and video data
class MultimodalDataset(Dataset):
    """Custom PyTorch dataset for multimodal inputs (text + video)."""
    def __init__(self, df_text, video_files, df_ground_truth, tokenizer):
        self.df_text = df_text  # Store the textual data DataFrame
        self.video_files = video_files  # Store the list of video file paths
        self.df_ground_truth = df_ground_truth  # Store the ground truth DataFrame
        self.tokenizer = tokenizer  # Store the tokenizer for text processing

    def __len__(self):
        return len(self.df_text)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        # Load textual data for the current index
        creative_data_title = self.df_text.iloc[idx]['creative_data_title']  # Get the title
        creative_data_description = self.df_text.iloc[idx]['creative_data_description']  # Get the description
        speech = self.df_text.iloc[idx]['speech']  # Get the speech text

        # Load video frames and extract features
        frames = self.load_video_frames(self.video_files[idx])  # Load video frames for the current video file
        video_features = self.extract_video_features(frames)  # Extract features from the video frames

        # Tokenize textual data
        inputs = self.tokenizer.encode_plus(
            creative_data_title,  # Input the title
            creative_data_description,  # Input the description
            speech,  # Input the speech text
            add_special_tokens=True,  # Add special tokens for BERT
            max_length=512,  # Maximum length of the tokenized input
            return_attention_mask=True,  # Return attention masks
            return_tensors='pt'  # Return PyTorch tensors
        )

        # Create multimodal input tensor by combining text and video features
        multimodal_input = torch.cat((inputs['input_ids'].flatten(), video_features), dim=0)  # Concatenate text and video features

        return multimodal_input, idx  # Return the multimodal input and the index

    def load_video_frames(self, video_file):
        """Load video frames from .mp4 file."""
        cap = cv2.VideoCapture(video_file)  # Open the video file
        frames = []  # Initialize an empty list to store frames
        while True:
            ret, frame = cap.read()  # Read a frame
            if not ret:  # If no frame is returned, break the loop
                break
            frames.append(frame)  # Add the frame to the list
        cap.release()  # Release the video capture object
        return frames  # Return the list of frames

    def extract_video_features(self, frames):
        """Extract video features using ResNet-50."""
        cnn = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)  # Load a pre-trained ResNet-50 model
        features = []  # Initialize an empty list to store features
        for frame in frames:  # Loop through each frame
            frame = cv2.resize(frame, (224, 224))  # Resize the frame to 224x224
            frame = frame / 255.0  # Normalize pixel values to [0, 1]
            frame = torch.tensor(frame).permute(2, 0, 1)  # Convert the frame to a tensor and permute dimensions
            features.append(cnn(frame.unsqueeze(0)).detach().numpy())  # Extract features and add to the list
        features = torch.tensor(features).mean(dim=0)  # Calculate the mean of the features
        return features  # Return the features

# Define Multimodal Model
class MultimodalModel(nn.Module):
    """Multimodal model combining BERT and ResNet-50."""
    def __init__(self, num_labels):
        super(MultimodalModel, self).__init__()
        self.bert_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)  # Load pre-trained BERT model
        self.video_fc = nn.Linear(2048, 128)  # Define a fully connected layer for video features
        self.classifier = nn.Linear(768 + 128, num_labels)  # Define a classifier that combines BERT and video features

    def forward(self, multimodal_input):
        textual_input = multimodal_input[:, :512]  # Extract textual input from multimodal input
        video_input = multimodal_input[:, 512:]  # Extract video input from multimodal input
        video_input = self.video_fc(video_input)  # Apply the fully connected layer to video features
        textual_output = self.bert_model(textual_input)[0]  # Get BERT output for textual input
        combined_input = torch.cat((textual_output[:, 0, :], video_input), dim=1)  # Concatenate textual and video features
        outputs = self.classifier(combined_input)  # Apply the classifier to the combined input
        return outputs  # Return the outputs

# Function to predict answers for all questions for a given dataset
def predict_answers(model, dataset, data_loader, device):
    """Predict answers using the trained multimodal model."""
    model.eval()  # Set model to evaluation mode
    results = []  # Initialize an empty list to store results
    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:  # Loop through batches
            multimodal_input, idx = batch  # Get multimodal input and index
            multimodal_input = multimodal_input.to(device)  # Move input to the device
            outputs = model(multimodal_input)  # Get model outputs
            _, predicted = torch.max(outputs, dim=1)  # Get predicted labels
            results.extend(predicted.cpu().numpy())  # Add predictions to results list
    return results  # Return the results

# Function to calculate evaluation metrics
def calculate_metrics(results_df, df_ground_truth):
    """Calculate evaluation metrics (F1 score, precision, recall)."""
    num_questions = len(df_ground_truth.columns) - 1  # Calculate the number of questions
    metrics = {}  # Initialize an empty dictionary to store metrics
    for i in range(1, num_questions + 1):  # Loop through each question
        question_name = f'question_{i}'  # Get the question name
        ground_truth = df_ground_truth[question_name].values  # Get ground truth values
        predictions = results_df[question_name].values  # Get predicted values
        f1 = f1_score(ground_truth, predictions, average='binary', pos_label='yes')  # Calculate F1 score
        precision = precision_score(ground_truth, predictions, pos_label='yes')  # Calculate precision
        recall = recall_score(ground_truth, predictions, pos_label='yes')  # Calculate recall
        metrics[question_name] = {'f1': f1, 'precision': precision, 'recall': recall}  # Store metrics in the dictionary
    return metrics  # Return the metrics

# Function to calculate agreement percentage
def calculate_agreement_percentage(results_df, df_ground_truth):
    """Calculate the agreement percentage between predictions and ground truth."""
    agreement = (results_df == df_ground_truth).mean(axis=0)  # Calculate mean agreement across all questions
    return agreement.mean() * 100  # Return overall agreement percentage

# Function to analyze videos that do not perform well with the classifier
def analyze_videos_performance(results_df, df_ground_truth):
    """Identify videos with low prediction performance."""
    differences = (results_df != df_ground_truth).sum(axis=1)  # Calculate differences between predictions and ground truth
    poorly_predicted_videos = differences[differences > differences.median()].index.tolist()  # Identify videos with above median differences
    return poorly_predicted_videos  # Return the list of poorly predicted videos

# Function to analyze reasons for inconsistent answers across questions
def analyze_inconsistent_answers(results_df, df_ground_truth):
    """Identify questions with inconsistent answers."""
    inconsistencies = (results_df != df_ground_truth).sum(axis=0)  # Calculate inconsistencies for each question
    inconsistent_questions = inconsistencies[inconsistencies > inconsistencies.median()].index.tolist()  # Identify questions with above median inconsistencies
    return inconsistent_questions  # Return the list of inconsistent questions

# Function to compare human coder responses with model predictions
def compare_human_coder_vs_model(results_df, df_ground_truth):
    """Compare human coder responses with model predictions."""
    comparison = (results_df == df_ground_truth).mean(axis=1)  # Calculate agreement for each video
    return comparison  # Return the comparison results

# Main function to execute the analysis
def main():
    """Main function to train the model and analyze results."""
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Load BERT tokenizer

    # Initialize dataset and dataloader for training
    dataset_train = MultimodalDataset(df_text, video_files, df_ground_truth, tokenizer)
    data_loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = MultimodalModel(num_labels=21).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader_train:
            multimodal_input, idx = batch
            multimodal_input = multimodal_input.to(device)
            labels = torch.tensor(df_ground_truth.iloc[idx, 1:].values).to(device)

            optimizer.zero_grad()
            outputs = model(multimodal_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(data_loader_train)}')

    # Define dataset and dataloader for prediction
    dataset_pred = MultimodalDataset(df_text, video_files, df_ground_truth, tokenizer)
    data_loader_pred = DataLoader(dataset_pred, batch_size=32, shuffle=False)

    # Predict answers for all questions
    predictions = predict_answers(model, dataset_pred, data_loader_pred, device)

    # Create results dataframe
    results_df = pd.DataFrame(predictions, columns=[f'question_{i}' for i in range(1, 22)])

    # Calculate evaluation metrics
    metrics = calculate_metrics(results_df, df_ground_truth)
    print('Evaluation Metrics:', metrics)

    # Calculate agreement percentage
    agreement_percentage = calculate_agreement_percentage(results_df, df_ground_truth)
    print('Agreement Percentage:', agreement_percentage)

    # Analyze videos that do not perform well with the classifier
    poorly_predicted_videos = analyze_videos_performance(results_df, df_ground_truth)
    print('Poorly Predicted Videos:', poorly_predicted_videos)

    # Analyze reasons for inconsistent answers across questions
    inconsistent_questions = analyze_inconsistent_answers(results_df, df_ground_truth)
    print('Inconsistent Questions:', inconsistent_questions)

    # Compare human coder responses with model predictions
    comparison_results = compare_human_coder_vs_model(results_df, df_ground_truth)
    print('Comparison of Human Coder Responses vs Model Predictions:', comparison_results)

if __name__ == "__main__":
    main()
