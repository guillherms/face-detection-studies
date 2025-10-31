import os
import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace


def detect_facial_expressions(video_path: str, output_path: str) -> None:
    """
    Detects facial expressions in an image using DeepFace.

    Args:
        image_path (str): Path to the image file.
    Returns:
        dict: A dictionary containing the detected facial expression and its confidence score.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for _ in tqdm(range(total_frames), desc="Processing video frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze the frame for facial expressions.
            # Note: Access DeepFace to see available models and actions.
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            # Iterar sobre cada face detectada pelo DeepFace
            for face in result:
                # Obter a caixa delimitadora da face
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                
                # Obter a emoção dominante
                dominant_emotion = face['dominant_emotion']

                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Escrever a emoção dominante acima da face
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Escrever o frame processado no vídeo de saída
            out.write(frame)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    except ValueError as ve:
        print(ve)
        return None
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = 'video/facial_expression.mp4'
    output_video_path = os.path.join(script_dir, 'output_video.mp4')  # Nome do vídeo de saída
    detect_facial_expressions(input_video_path, output_video_path)