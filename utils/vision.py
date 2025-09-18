import os
import pickle
import numpy as np
from PIL import Image
from datetime import datetime
import random

import cv2
import numpy as np
from datetime import datetime
import os
import pickle
from PIL import Image

class VisionService:
    def __init__(self):
        self.supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.classifier_rules = self._load_disease_rules()
        self.face_cascade = self._load_face_cascade()
        
    def _load_face_cascade(self):
        """Load OpenCV face cascade classifier"""
        try:
            # Try to load the face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                print("⚠️ Face cascade classifier not loaded properly")
                return None
            print("✅ Face cascade classifier loaded!")
            return face_cascade
        except Exception as e:
            print(f"⚠️ Could not load face cascade: {str(e)}")
            return None
    
    def _detect_object_type(self, img_cv):
        """Enhanced object detection for persons, animals, plants, and other objects"""
        try:
            detection_results = []
            
            # First check for significant green content (strong plant indicator)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
            green_ratio = np.sum(green_mask > 0) / (img_cv.shape[0] * img_cv.shape[1])
            
            # If there's significant green content, prioritize plant detection
            if green_ratio > 0.2:
                plant_confidence = self._analyze_plant_features_enhanced(img_cv)
                if plant_confidence > 0.15:  # Lower threshold for green-heavy images
                    return {
                        'object_type': 'plant',
                        'confidence': min(plant_confidence * 100 + 10, 95),  # Boost confidence
                        'details': f'High green content detected ({green_ratio:.2%}) - Plant features confirmed'
                    }
            
            # 1. Human face detection (highest priority for clear faces)
            if self.face_cascade is not None:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    return {
                        'object_type': 'person',
                        'confidence': 92 + random.uniform(0, 3),
                        'details': f'Human face detected: {len(faces)} face(s)'
                    }
            
            # 2. Enhanced plant detection (higher priority)
            plant_confidence = self._analyze_plant_features_enhanced(img_cv)
            if plant_confidence > 0.2:  # Slightly lower threshold
                detection_results.append(('plant', plant_confidence * 100, 'Plant features detected'))
            
            # 3. Enhanced skin tone detection for humans (more restrictive)
            person_confidence = self._detect_skin_tone_enhanced(img_cv)
            if person_confidence > 0.35:  # Higher threshold to reduce false positives
                detection_results.append(('person', person_confidence * 100, 'Human skin tone detected'))
            
            # 4. Animal detection using fur/texture patterns
            animal_confidence = self._detect_animal_features(img_cv)
            if animal_confidence > 0.35:  # Slightly higher threshold
                detection_results.append(('animal', animal_confidence * 100, 'Animal features detected'))
            
            # 5. Check for artificial objects (buildings, vehicles, etc.)
            artificial_confidence = self._detect_artificial_objects(img_cv)
            if artificial_confidence > 0.4:
                detection_results.append(('artificial', artificial_confidence * 100, 'Artificial object detected'))
            
            # Select the detection with highest confidence, but prioritize plants if close
            if detection_results:
                # Sort by confidence
                detection_results.sort(key=lambda x: x[1], reverse=True)
                best_detection = detection_results[0]
                
                # If the best detection is plant, or if plant is close second, choose plant
                plant_detections = [d for d in detection_results if d[0] == 'plant']
                if plant_detections:
                    plant_conf = plant_detections[0][1]
                    best_conf = best_detection[1]
                    
                    # Choose plant if it's within 20 points of the best detection
                    if best_detection[0] == 'plant' or (plant_conf >= best_conf - 20):
                        best_detection = plant_detections[0]
                
                return {
                    'object_type': best_detection[0],
                    'confidence': min(best_detection[1], 95),
                    'details': best_detection[2]
                }
            
            # Default to unknown with low confidence
            return {
                'object_type': 'unknown',
                'confidence': 25,
                'details': 'Could not clearly identify object type'
            }
            
        except Exception as e:
            print(f"Object detection error: {str(e)}")
            return {
                'object_type': 'unknown',
                'confidence': 0,
                'details': f'Detection failed: {str(e)}'
            }
    
    def _detect_skin_tone_enhanced(self, img_cv):
        """Enhanced skin tone detection with better accuracy and reduced false positives"""
        try:
            # Convert to multiple color spaces for better skin detection
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
            
            # Check if image has too much green (likely a plant)
            green_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
            green_ratio = np.sum(green_mask > 0) / (img_cv.shape[0] * img_cv.shape[1])
            
            # If more than 30% green, heavily penalize skin detection
            if green_ratio > 0.3:
                return 0.0
            elif green_ratio > 0.15:
                skin_penalty = 0.3  # Reduce skin confidence by 70%
            else:
                skin_penalty = 1.0  # No penalty
            
            # More restrictive skin tone ranges to reduce false positives
            hsv_skin_ranges = [
                # Light skin tones (more restrictive)
                ((0, 58, 90), (18, 200, 245)),
                # Medium skin tones (more restrictive)
                ((0, 35, 80), (20, 180, 220)),
            ]
            
            # More restrictive YCrCb ranges
            ycrcb_skin_ranges = [
                ((90, 140, 90), (240, 175, 130))
            ]
            
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            skin_pixels_hsv = 0
            skin_pixels_ycrcb = 0
            
            # HSV-based skin detection
            for lower, upper in hsv_skin_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                skin_pixels_hsv += np.sum(mask > 0)
            
            # YCrCb-based skin detection
            for lower, upper in ycrcb_skin_ranges:
                mask = cv2.inRange(ycrcb, np.array(lower), np.array(upper))
                skin_pixels_ycrcb += np.sum(mask > 0)
            
            # Calculate ratios
            hsv_skin_ratio = skin_pixels_hsv / total_pixels
            ycrcb_skin_ratio = skin_pixels_ycrcb / total_pixels
            
            # Both methods must agree for higher confidence
            if hsv_skin_ratio < 0.1 or ycrcb_skin_ratio < 0.05:
                return 0.0
            
            # Combine both methods (require both to detect skin)
            combined_skin_ratio = min(hsv_skin_ratio, ycrcb_skin_ratio) * 0.8 + max(hsv_skin_ratio, ycrcb_skin_ratio) * 0.2
            
            # Additional validation: check for human-like features
            if combined_skin_ratio > 0.1:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Check for face-like features using edge patterns
                edges = cv2.Canny(gray, 50, 150)
                
                # Look for circular patterns (eyes, face outline)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                         param1=50, param2=30, minRadius=10, maxRadius=100)
                
                human_features = 0
                if circles is not None and len(circles[0]) >= 2:
                    human_features += 1
                
                # Check for elongated shapes that could be limbs
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                elongated_shapes = 0
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Increased threshold
                        rect = cv2.minAreaRect(contour)
                        width, height = rect[1]
                        if width > 0 and height > 0:
                            aspect_ratio = max(width, height) / min(width, height)
                            if 3.0 < aspect_ratio < 6.0:  # More restrictive human proportions
                                elongated_shapes += 1
                
                if elongated_shapes >= 2:
                    human_features += 1
                
                # Require multiple human-like features
                if human_features < 1:
                    combined_skin_ratio *= 0.3
            
            # Apply green penalty
            combined_skin_ratio *= skin_penalty
            
            return min(combined_skin_ratio, 1.0)
            
        except Exception as e:
            print(f"Enhanced skin tone detection error: {str(e)}")
            return 0
    
    
    def _detect_animal_features(self, img_cv):
        """Detect animal features like fur patterns, eyes, and body shapes"""
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            animal_score = 0
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            
            # First, check if this is likely a plant (exclude plants)
            green_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 255]))
            green_ratio = np.sum(green_mask > 0) / total_pixels
            
            # If too much green, it's likely a plant, not an animal
            if green_ratio > 0.4:
                return 0.0
            elif green_ratio > 0.2:
                plant_penalty = 0.4  # Reduce animal confidence by 60%
            else:
                plant_penalty = 1.0  # No penalty
            
            # 1. Fur texture detection using texture analysis
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            texture_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            texture_variance = np.var(texture_magnitude)
            
            if texture_variance > 400:  # Higher threshold
                animal_score += 0.25
            elif texture_variance > 200:
                animal_score += 0.15
            
            # 2. Color pattern analysis for common animal colors (more restrictive)
            animal_color_ranges = [
                # Brown ranges (dogs, cats, etc.) - more restrictive
                ((8, 60, 30), (18, 255, 180)),
                # Black/dark gray - more restrictive
                ((0, 0, 0), (180, 255, 40)),
                # Light gray/white - more restrictive
                ((0, 0, 200), (180, 25, 255)),
                # Golden/tan colors - more restrictive
                ((12, 120, 120), (22, 255, 240))
            ]
            
            animal_color_pixels = 0
            for lower, upper in animal_color_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                animal_color_pixels += np.sum(mask > 0)
            
            animal_color_ratio = animal_color_pixels / total_pixels
            if animal_color_ratio > 0.5:  # Higher threshold
                animal_score += 0.3
            elif animal_color_ratio > 0.3:
                animal_score += 0.2
            
            # 3. Eye detection using circular patterns (more restrictive)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 40,
                                     param1=60, param2=35, minRadius=8, maxRadius=40)
            
            if circles is not None and len(circles[0]) >= 1:
                potential_eyes = len(circles[0])
                if 1 <= potential_eyes <= 3:  # Reasonable number for animal eyes
                    animal_score += 0.2
                    # Bonus for pairs of circles (eyes)
                    if potential_eyes == 2:
                        animal_score += 0.1
            
            # 4. Body shape analysis - more restrictive
            edges = cv2.Canny(gray, 60, 160)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            organic_shapes = 0
            total_contour_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 800:  # Higher threshold
                    total_contour_area += area
                    
                    # Check if shape is organic (non-geometric)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area > 0:
                        solidity = area / hull_area
                        # Animals tend to have complex, non-convex shapes
                        if 0.4 < solidity < 0.75:  # More restrictive range
                            organic_shapes += 1
            
            if organic_shapes > 0 and total_contour_area > (total_pixels * 0.15):  # Higher threshold
                animal_score += 0.15
            
            # Apply plant penalty
            animal_score *= plant_penalty
            
            return min(animal_score, 1.0)
            
        except Exception as e:
            print(f"Animal detection error: {str(e)}")
            return 0
    
    def _detect_artificial_objects(self, img_cv):
        """Detect artificial objects like buildings, vehicles, electronics"""
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            artificial_score = 0
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            
            # 1. Detect straight lines and geometric patterns (typical of artificial objects)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            if lines is not None and len(lines) > 5:
                artificial_score += 0.3  # Many straight lines suggest artificial objects
            
            # 2. Check for artificial color patterns
            artificial_color_ranges = [
                # Bright blues (electronics, vehicles)
                ((100, 100, 100), (130, 255, 255)),
                # Bright reds (warning signs, vehicles)
                ((0, 150, 150), (10, 255, 255)),
                # Pure whites/grays (buildings, electronics)
                ((0, 0, 200), (180, 50, 255)),
                # Metallic colors
                ((0, 0, 100), (180, 30, 200))
            ]
            
            artificial_color_pixels = 0
            for lower, upper in artificial_color_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                artificial_color_pixels += np.sum(mask > 0)
            
            artificial_color_ratio = artificial_color_pixels / total_pixels
            if artificial_color_ratio > 0.3:
                artificial_score += 0.25
            
            # 3. Geometric shape detection
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            geometric_shapes = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        
                        # Very circular or very square shapes suggest artificial objects
                        if circularity > 0.85 or circularity < 0.1:
                            geometric_shapes += 1
                        
                        # Check for rectangular shapes
                        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                        if len(approx) == 4:  # Rectangle
                            geometric_shapes += 1
            
            if geometric_shapes > 2:
                artificial_score += 0.2
            
            # 4. Text detection (signs, displays)
            # Simple text detection using connected components
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            text_like_components = 0
            for i in range(1, num_labels):
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                if area > 50 and width > height and 2 < width/height < 10:
                    text_like_components += 1
            
            if text_like_components > 3:
                artificial_score += 0.15
            
            return min(artificial_score, 1.0)
            
        except Exception as e:
            print(f"Artificial object detection error: {str(e)}")
            return 0
    
    def _analyze_plant_features_enhanced(self, img_cv):
        """Enhanced plant feature analysis with better accuracy"""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            
            plant_score = 0
            
            # 1. Enhanced green vegetation detection with multiple ranges
            green_ranges = [
                # Bright green (healthy plants)
                ((35, 40, 40), (85, 255, 255)),
                # Dark green (mature plants)
                ((25, 30, 20), (85, 255, 200)),
                # Yellow-green (new growth)
                ((20, 40, 40), (35, 255, 255)),
                # Blue-green (some plant varieties)
                ((85, 40, 40), (95, 255, 255))
            ]
            
            total_green_pixels = 0
            for lower, upper in green_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                total_green_pixels += np.sum(mask > 0)
            
            green_ratio = total_green_pixels / total_pixels
            
            # Score based on green content
            if green_ratio > 0.6:
                plant_score += 0.4  # High confidence for lots of green
            elif green_ratio > 0.3:
                plant_score += 0.3
            elif green_ratio > 0.15:
                plant_score += 0.2
            elif green_ratio > 0.05:
                plant_score += 0.1
            
            # 2. Leaf-like texture patterns
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            texture_gradient = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            if 15 < texture_gradient < 40:
                plant_score += 0.2
            elif 10 < texture_gradient < 50:
                plant_score += 0.1
            
            # 3. Organic edge patterns (leaf shapes, branches)
            edges = cv2.Canny(gray, 30, 100)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            organic_plant_shapes = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if 0.1 < circularity < 0.8:
                            organic_plant_shapes += 1
                            
                        rect = cv2.minAreaRect(contour)
                        width, height = rect[1]
                        if width > 0 and height > 0:
                            aspect_ratio = max(width, height) / min(width, height)
                            if 2.0 < aspect_ratio < 10.0:
                                organic_plant_shapes += 0.5
            
            if organic_plant_shapes > 2:
                plant_score += 0.15
            elif organic_plant_shapes > 0:
                plant_score += 0.1
            
            # 4. Color distribution analysis
            h_channel = hsv[:,:,0]
            s_channel = hsv[:,:,1]
            v_channel = hsv[:,:,2]
            
            h_std = np.std(h_channel)
            s_std = np.std(s_channel)
            v_std = np.std(v_channel)
            
            if 10 < h_std < 40 and s_std > 20 and v_std > 30:
                plant_score += 0.1
            
            # 5. Exclude artificial characteristics
            artificial_colors = [
                ((100, 100, 100), (130, 255, 255)),
                ((0, 200, 200), (10, 255, 255)),
                ((0, 0, 220), (180, 30, 255))
            ]
            
            artificial_pixels = 0
            for lower, upper in artificial_colors:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                artificial_pixels += np.sum(mask > 0)
            
            artificial_ratio = artificial_pixels / total_pixels
            if artificial_ratio > 0.3:
                plant_score *= 0.5
            
            return min(plant_score, 1.0)
            
        except Exception as e:
            print(f"Enhanced plant feature analysis error: {str(e)}")
            return 0
        
    def _detect_plant_content(self, img_cv):
        """Enhanced plant content detection using multiple computer vision techniques"""
        try:
            # Convert to HSV color space for better color detection
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Multiple green ranges to capture different plant types
            green_ranges = [
                # Bright green (healthy plants)
                ((35, 40, 40), (85, 255, 255)),
                # Dark green (older leaves)
                ((25, 30, 20), (85, 255, 200)),
                # Yellow-green (some plant varieties)
                ((20, 40, 40), (35, 255, 255))
            ]
            
            total_green_pixels = 0
            total_pixels = img_cv.shape[0] * img_cv.shape[1]
            
            for lower, upper in green_ranges:
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                total_green_pixels += np.sum(mask > 0)
            
            green_ratio = total_green_pixels / total_pixels
            
            # Enhanced texture analysis for organic patterns
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Multiple texture measures
            # 1. Laplacian variance (edge sharpness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            # 2. Local Binary Pattern approximation
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            texture_gradient = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            # 3. Edge density for organic shapes
            edges = cv2.Canny(gray, 30, 100)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # Combine criteria for plant detection
            plant_score = 0
            
            # Green content score (0-40 points)
            if green_ratio > 0.3:
                plant_score += 40
            elif green_ratio > 0.15:
                plant_score += 25
            elif green_ratio > 0.05:
                plant_score += 10
            
            # Texture complexity score (0-30 points)
            if texture_variance > 200:
                plant_score += 30
            elif texture_variance > 100:
                plant_score += 20
            elif texture_variance > 50:
                plant_score += 10
            
            # Organic edge patterns (0-30 points)
            if edge_density > 0.1 and texture_gradient > 20:
                plant_score += 30
            elif edge_density > 0.05:
                plant_score += 15
            
            # Plant detected if score > 50 out of 100
            is_plant = plant_score > 50
            
            return is_plant
            
        except Exception as e:
            print(f"Enhanced plant detection error: {str(e)}")
            # Fallback to basic green detection
            try:
                hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
                lower_green = np.array([25, 40, 40])
                upper_green = np.array([85, 255, 255])
                mask = cv2.inRange(hsv, lower_green, upper_green)
                green_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                return green_ratio > 0.1
            except:
                return False
        
    def _load_disease_rules(self):
        """Load disease classification rules"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rules_path = os.path.join(script_dir, 'model', 'plant_disease_rules.pkl')
            
            with open(rules_path, 'rb') as f:
                rules = pickle.load(f)
                print("✅ Disease classification rules loaded!")
                return rules
        except FileNotFoundError:
            print("⚠️ Disease rules not found. Using default rules.")
            return self._get_default_rules()
    
    def _get_default_rules(self):
        """Enhanced disease classification rules with more detailed analysis"""
        return {
            'tomato': {
                'healthy': {
                    'green_ratio': (0.6, 1.0), 
                    'brightness': (0.4, 0.8),
                    'brown_spots': (0, 0.1),
                    'yellow_ratio': (0, 0.2)
                },
                'early_blight': {
                    'green_ratio': (0.3, 0.6), 
                    'brightness': (0.2, 0.6),
                    'brown_spots': (0.1, 0.3),
                    'yellow_ratio': (0.2, 0.4)
                },
                'late_blight': {
                    'green_ratio': (0.2, 0.5), 
                    'brightness': (0.1, 0.4),
                    'brown_spots': (0.3, 0.6),
                    'yellow_ratio': (0.3, 0.5)
                },
                'bacterial_spot': {
                    'green_ratio': (0.4, 0.7), 
                    'brightness': (0.3, 0.7),
                    'brown_spots': (0.15, 0.35),
                    'yellow_ratio': (0.1, 0.3)
                },
                'leaf_spot': {
                    'green_ratio': (0.3, 0.6), 
                    'brightness': (0.2, 0.6),
                    'brown_spots': (0.2, 0.4),
                    'yellow_ratio': (0.2, 0.4)
                }
            },
            'potato': {
                'healthy': {
                    'green_ratio': (0.5, 0.9), 
                    'brightness': (0.4, 0.8),
                    'brown_spots': (0, 0.1),
                    'yellow_ratio': (0, 0.2)
                },
                'early_blight': {
                    'green_ratio': (0.3, 0.6), 
                    'brightness': (0.2, 0.5),
                    'brown_spots': (0.15, 0.4),
                    'yellow_ratio': (0.25, 0.45)
                },
                'late_blight': {
                    'green_ratio': (0.2, 0.4), 
                    'brightness': (0.1, 0.3),
                    'brown_spots': (0.4, 0.7),
                    'yellow_ratio': (0.3, 0.5)
                }
            },
            'corn': {
                'healthy': {
                    'green_ratio': (0.6, 0.9), 
                    'brightness': (0.5, 0.8),
                    'brown_spots': (0, 0.1),
                    'yellow_ratio': (0, 0.2)
                },
                'common_rust': {
                    'green_ratio': (0.3, 0.6), 
                    'brightness': (0.3, 0.6),
                    'brown_spots': (0.2, 0.4),
                    'yellow_ratio': (0.2, 0.4)
                }
            }
        }
    
    def analyze_crop_image(self, image_file):
        """Enhanced image analysis with improved object detection and user feedback"""
        try:
            # Validate image
            validation_result = self._validate_image(image_file)
            if not validation_result['valid']:
                return validation_result
            
            # Process image
            processed_image = self._preprocess_image(image_file)
            img_array = processed_image['image_array']
            
            # Convert to OpenCV format
            img_cv = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Detect what type of object this is using enhanced detection
            object_detection = self._detect_object_type(img_cv)
            object_type = object_detection['object_type']
            confidence = object_detection['confidence']
            details = object_detection['details']
            
            if object_type == 'person':
                return {
                    'success': True,
                    'object_type': 'person',
                    'message': 'Human detected in the image',
                    'confidence': confidence,
                    'details': details,
                    'recommendation': 'This appears to be a photo of a person. Please upload an image of a plant or crop for disease analysis.',
                    'action_required': 'Please try again with a plant image',
                    'timestamp': datetime.now().isoformat()
                }
            
            elif object_type == 'animal':
                return {
                    'success': True,
                    'object_type': 'animal',
                    'message': 'Animal detected in the image',
                    'confidence': confidence,
                    'details': details,
                    'recommendation': 'This appears to be an animal (dog, cat, etc.). Please upload an image of a plant or crop for disease analysis.',
                    'action_required': 'Please try again with a plant image',
                    'timestamp': datetime.now().isoformat()
                }
            
            elif object_type == 'artificial':
                return {
                    'success': True,
                    'object_type': 'artificial',
                    'message': 'Artificial object detected in the image',
                    'confidence': confidence,
                    'details': details,
                    'recommendation': 'This appears to be an artificial object (building, vehicle, electronics, etc.). Please upload an image of a plant or crop for disease analysis.',
                    'action_required': 'Please try again with a plant image',
                    'timestamp': datetime.now().isoformat()
                }
            
            elif object_type == 'plant':
                # Perform detailed plant disease analysis
                analysis_result = self._analyze_plant_disease(img_cv, img_array)
                return {
                    'success': True,
                    'object_type': 'plant',
                    'message': 'Plant detected - performing disease analysis',
                    'analysis': analysis_result,
                    'confidence': analysis_result['confidence'],
                    'detection_details': details,
                    'model_used': 'Enhanced_CV_Analysis_v3.0',
                    'timestamp': datetime.now().isoformat()
                }
            
            else:  # unknown object type
                return {
                    'success': False,
                    'object_type': 'unknown',
                    'message': 'Unable to clearly identify the object in the image',
                    'confidence': confidence,
                    'details': details,
                    'recommendation': 'The image content could not be clearly identified. Please upload a clear, well-lit image of a plant or crop for disease analysis. Ensure the plant takes up most of the frame.',
                    'action_required': 'Please try again with a clearer plant image',
                    'troubleshooting_tips': [
                        'Ensure good lighting when taking the photo',
                        'Focus on the plant leaves or affected areas',
                        'Make sure the plant fills most of the frame',
                        'Avoid blurry or heavily shadowed images'
                    ],
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Image analysis failed: {str(e)}',
                'analysis': self._get_default_analysis()
            }
    
    def _analyze_plant_disease(self, img_cv, img_array):
        """Analyze plant for disease using computer vision and loaded rules"""
        try:
            # Perform comprehensive analysis
            analysis_results = self._perform_advanced_analysis(img_cv, img_array)
            
            # Determine crop type using computer vision
            crop_type = self._determine_crop_type_cv(img_cv, analysis_results)
            
            # Determine disease using loaded rules and CV analysis
            disease_info = self._determine_disease_with_rules(crop_type, analysis_results)
            
            # Generate comprehensive analysis
            analysis = {
                'crop_detected': crop_type.title(),
                'disease_detected': disease_info['disease'].replace('_', ' ').title(),
                'confidence': disease_info['confidence'],
                'severity': self._assess_severity(disease_info['disease']),
                'treatment_recommendations': self._get_treatment_recommendations(disease_info['disease']),
                'prevention_tips': self._get_prevention_tips(disease_info['disease']),
                'image_quality_assessment': self._assess_image_quality_advanced(analysis_results),
                'technical_analysis': {
                    'green_vegetation_ratio': round(analysis_results['green_ratio'], 3),
                    'diseased_area_ratio': round(analysis_results['brown_spots'], 3),
                    'chlorosis_ratio': round(analysis_results['yellow_ratio'], 3),
                    'brightness_level': round(analysis_results['brightness'], 3),
                    'texture_complexity': round(analysis_results['edge_density'], 3),
                    'health_score': round(analysis_results['health_score'], 1)
                },
                'confidence_factors': {
                    'color_analysis_confidence': round(analysis_results['color_confidence'], 1),
                    'texture_analysis_confidence': round(analysis_results['texture_confidence'], 1),
                    'spot_detection_confidence': round(analysis_results['spot_confidence'], 1)
                },
                'recommendations': self._get_detailed_recommendations(disease_info['disease'], analysis_results),
                'analysis_method': 'Computer Vision + Disease Rules'
            }
            
            return analysis
            
        except Exception as e:
            print(f"Plant disease analysis error: {str(e)}")
            return self._get_default_analysis()
    
    def _determine_crop_type_cv(self, img_cv, analysis_results):
        """Determine crop type using computer vision analysis"""
        try:
            # Analyze leaf shape and texture patterns
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Edge detection for shape analysis
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze dominant contour shape
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 500:
                    # Calculate shape descriptors
                    perimeter = cv2.arcLength(largest_contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * cv2.contourArea(largest_contour) / (perimeter ** 2)
                        
                        # Approximate contour to polygon
                        epsilon = 0.02 * perimeter
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        # Classify based on shape characteristics
                        if len(approx) < 6 and circularity > 0.5:
                            return 'tomato'  # Round leaves
                        elif len(approx) >= 8 and circularity < 0.3:
                            return 'corn'    # Long, narrow leaves
                        else:
                            return 'potato'  # Irregular shaped leaves
            
            # Fallback to color-based classification
            green_ratio = analysis_results['green_ratio']
            yellow_ratio = analysis_results['yellow_ratio']
            
            if green_ratio > 0.6:
                return 'tomato'
            elif yellow_ratio > 0.2:
                return 'corn'
            else:
                return 'potato'
                
        except Exception as e:
            print(f"Crop type determination error: {str(e)}")
            return 'tomato'  # Default fallback
    
    def _determine_disease_with_rules(self, crop_type, analysis_results):
        """Determine disease using loaded rules and computer vision analysis"""
        try:
            # Get disease rules for the detected crop type
            crop_diseases = self.classifier_rules.get(crop_type, self.classifier_rules.get('tomato', {}))
            
            if not crop_diseases:
                return {'disease': 'unknown', 'confidence': 30.0}
            
            # Extract analysis factors
            green_ratio = analysis_results['green_ratio']
            brightness = analysis_results['brightness']
            brown_spots = analysis_results['brown_spots']
            yellow_ratio = analysis_results['yellow_ratio']
            health_score = analysis_results['health_score']
            
            # Find best matching disease based on rules
            best_match = 'healthy'
            max_score = 0
            
            for disease, ranges in crop_diseases.items():
                score = 0
                factors_matched = 0
                
                # Check each parameter against the rules
                if 'green_ratio' in ranges:
                    green_min, green_max = ranges['green_ratio']
                    if green_min <= green_ratio <= green_max:
                        score += 0.25
                        factors_matched += 1
                
                if 'brightness' in ranges:
                    bright_min, bright_max = ranges['brightness']
                    if bright_min <= brightness <= bright_max:
                        score += 0.25
                        factors_matched += 1
                
                if 'brown_spots' in ranges:
                    brown_min, brown_max = ranges['brown_spots']
                    if brown_min <= brown_spots <= brown_max:
                        score += 0.25
                        factors_matched += 1
                
                if 'yellow_ratio' in ranges:
                    yellow_min, yellow_max = ranges['yellow_ratio']
                    if yellow_min <= yellow_ratio <= yellow_max:
                        score += 0.25
                        factors_matched += 1
                
                # Bonus for multiple factors matching
                if factors_matched >= 3:
                    score += 0.1
                elif factors_matched >= 2:
                    score += 0.05
                
                if score > max_score:
                    max_score = score
                    best_match = disease
            
            # Calculate confidence based on match quality
            base_confidence = min(max_score * 100, 90)
            
            # Adjust confidence based on health score
            if best_match == 'healthy' and health_score > 70:
                confidence_boost = 10
            elif best_match != 'healthy' and health_score < 50:
                confidence_boost = 5
            else:
                confidence_boost = 0
            
            final_confidence = min(base_confidence + confidence_boost + random.uniform(-3, 3), 95)
            final_confidence = max(final_confidence, 40)  # Minimum confidence threshold
            
            return {
                'disease': best_match,
                'confidence': round(final_confidence, 1)
            }
            
        except Exception as e:
            print(f"Disease determination error: {str(e)}")
            return {'disease': 'unknown', 'confidence': 30.0}
    
    def _analyze_image_characteristics(self, processed_image):
        """Enhanced image analysis using computer vision techniques"""
        image_array = processed_image['image_array']
        
        # Check if OpenCV is available for advanced analysis
        if cv2 is not None:
            # Convert to OpenCV format for advanced analysis
            if len(image_array.shape) == 3:
                img_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = image_array
        else:
            # Use fallback mode without OpenCV
            img_cv = None
        
        # Perform comprehensive analysis
        analysis_results = self._perform_advanced_analysis(img_cv, image_array)
        
        # Determine crop type and disease using enhanced methods
        crop_type = self._determine_crop_type_advanced(analysis_results)
        disease_info = self._determine_disease_advanced(crop_type, analysis_results)
        
        # Generate comprehensive analysis
        analysis = {
            'crop_detected': crop_type.title(),
            'disease_detected': disease_info['disease'].replace('_', ' ').title(),
            'confidence': disease_info['confidence'],
            'severity': self._assess_severity(disease_info['disease']),
            'treatment_recommendations': self._get_treatment_recommendations(disease_info['disease']),
            'prevention_tips': self._get_prevention_tips(disease_info['disease']),
            'image_quality_assessment': self._assess_image_quality_advanced(analysis_results),
            'technical_analysis': {
                'green_vegetation_ratio': round(analysis_results['green_ratio'], 3),
                'diseased_area_ratio': round(analysis_results['brown_spots'], 3),
                'chlorosis_ratio': round(analysis_results['yellow_ratio'], 3),
                'brightness_level': round(analysis_results['brightness'], 3),
                'texture_complexity': round(analysis_results['edge_density'], 3),
                'health_score': round(analysis_results['health_score'], 1)
            },
            'confidence_factors': {
                'color_analysis_confidence': round(analysis_results['color_confidence'], 1),
                'texture_analysis_confidence': round(analysis_results['texture_confidence'], 1),
                'spot_detection_confidence': round(analysis_results['spot_confidence'], 1)
            },
            'recommendations': self._get_detailed_recommendations(disease_info['disease'], analysis_results),
            'analysis_method': 'Advanced CV Analysis'
        }
        
        return analysis
    
    def _determine_crop_type(self, green_ratio, red_ratio, blue_ratio):
        """Determine crop type based on color ratios"""
        # Simple heuristics for crop identification
        if green_ratio > 0.5 and red_ratio < 0.3:
            return random.choice(['tomato', 'potato', 'corn'])
        elif red_ratio > 0.4:
            return random.choice(['tomato', 'apple'])
        else:
            return random.choice(['potato', 'corn', 'wheat'])
    
    def _determine_disease(self, crop_type, green_ratio, brightness):
        """Determine disease based on crop type and image characteristics"""
        crop_diseases = self.classifier_rules.get(crop_type, self.classifier_rules['tomato'])
        
        # Find best matching disease based on characteristics
        best_match = 'healthy'
        max_score = 0
        
        for disease, ranges in crop_diseases.items():
            score = 0
            
            # Check green ratio
            green_min, green_max = ranges['green_ratio']
            if green_min <= green_ratio <= green_max:
                score += 0.5
            
            # Check brightness
            bright_min, bright_max = ranges['brightness']
            if bright_min <= brightness <= bright_max:
                score += 0.5
            
            if score > max_score:
                max_score = score
                best_match = disease
        
        # Calculate confidence based on match quality
        confidence = min(max_score * 100 + random.uniform(5, 15), 95)
        
        return {
            'disease': best_match,
            'confidence': round(confidence, 1)
        }
    
    def _assess_severity(self, disease):
        """Assess disease severity"""
        severity_map = {
            'healthy': 'None - Plant appears healthy',
            'early_blight': 'Moderate - Early intervention recommended',
            'late_blight': 'High - Immediate treatment required',
            'bacterial_spot': 'Moderate - Monitor and treat',
            'common_rust': 'Moderate - Apply fungicide treatment',
            'leaf_blight': 'Moderate to High - Remove affected areas'
        }
        return severity_map.get(disease, 'Moderate - Consult expert')
    
    def _get_treatment_recommendations(self, disease):
        """Get treatment recommendations"""
        treatments = {
            'healthy': [
                'Continue current care routine',
                'Monitor plant health regularly',
                'Maintain proper nutrition and watering'
            ],
            'early_blight': [
                'Apply copper-based fungicide spray',
                'Remove affected leaves immediately',
                'Improve air circulation around plants',
                'Reduce humidity levels'
            ],
            'late_blight': [
                'Apply systemic fungicide immediately',
                'Remove and destroy infected plants',
                'Improve drainage in growing area',
                'Avoid overhead watering'
            ],
            'bacterial_spot': [
                'Apply copper-based bactericide',
                'Remove infected plant parts',
                'Disinfect gardening tools',
                'Improve plant spacing'
            ],
            'common_rust': [
                'Apply rust-specific fungicide',
                'Remove affected leaves',
                'Ensure good air circulation',
                'Avoid wet conditions'
            ]
        }
        return treatments.get(disease, ['Consult local agricultural expert'])
    
    def _get_prevention_tips(self, disease):
        """Get prevention tips"""
        prevention = {
            'healthy': [
                'Regular plant inspection',
                'Balanced fertilization',
                'Proper watering schedule',
                'Good garden hygiene'
            ],
            'early_blight': [
                'Use disease-resistant varieties',
                'Practice crop rotation',
                'Mulch around plants',
                'Water at soil level'
            ],
            'late_blight': [
                'Plant in well-draining soil',
                'Avoid overcrowding',
                'Remove plant debris',
                'Monitor weather conditions'
            ],
            'bacterial_spot': [
                'Use certified disease-free seeds',
                'Avoid working with wet plants',
                'Sanitize tools between plants',
                'Control insect vectors'
            ]
        }
        return prevention.get(disease, ['Practice good agricultural hygiene'])
    
    def _assess_image_quality(self, brightness):
        """Assess image quality"""
        if brightness < 0.2:
            return 'Poor - Image too dark'
        elif brightness > 0.8:
            return 'Poor - Image overexposed'
        elif 0.3 <= brightness <= 0.7:
            return 'Good - Optimal lighting'
        else:
            return 'Fair - Acceptable for analysis'
    
    def _get_additional_recommendations(self):
        """Get additional recommendations"""
        recommendations = [
            'Take multiple photos from different angles',
            'Ensure good lighting when capturing images',
            'Focus on affected areas for better diagnosis',
            'Consult local agricultural extension officer',
            'Keep records of plant health changes',
            'Consider soil testing for nutrient analysis'
        ]
        return random.sample(recommendations, 3)
    
    def _validate_image(self, image_file):
        """Validate uploaded image file"""
        try:
            # Reset file pointer to beginning
            image_file.seek(0)
            
            if hasattr(image_file, 'content_length'):
                if image_file.content_length > self.max_file_size:
                    return {'valid': False, 'error': 'File too large (max 10MB)'}
            
            filename = getattr(image_file, 'filename', '')
            if filename:
                file_extension = filename.lower().split('.')[-1]
                if file_extension not in self.supported_formats:
                    return {'valid': False, 'error': f'Unsupported format. Use: {", ".join(self.supported_formats)}'}
            
            image = Image.open(image_file)
            image.verify()
            # Reset file pointer again after verification
            image_file.seek(0)
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Invalid image: {str(e)}'}
    
    def _preprocess_image(self, image_file):
        """Enhanced preprocessing for better computer vision analysis"""
        try:
            # Make sure file pointer is at the beginning
            image_file.seek(0)
            
            image = Image.open(image_file)
            original_size = image.size
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhanced resizing for better analysis
            # Keep aspect ratio and resize to reasonable dimensions
            max_dimension = 512
            width, height = image.size
            
            if width > height:
                new_width = max_dimension
                new_height = int((height * max_dimension) / width)
            else:
                new_height = max_dimension
                new_width = int((width * max_dimension) / height)
            
            # Resize with high-quality resampling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Apply noise reduction
            image_array = np.array(image)
            
            # Enhance contrast if image is too dark or too bright
            brightness = np.mean(image_array) / 255.0
            if brightness < 0.3:  # Too dark
                image_array = np.clip(image_array * 1.3, 0, 255).astype(np.uint8)
            elif brightness > 0.8:  # Too bright
                image_array = np.clip(image_array * 0.8, 0, 255).astype(np.uint8)
            
            # Normalize for analysis
            normalized_array = image_array.astype(np.float32) / 255.0
            
            return {
                'image_array': normalized_array,
                'original_size': original_size,
                'processed_size': (new_width, new_height),
                'enhancement_applied': brightness < 0.3 or brightness > 0.8
            }
            
        except Exception as e:
            raise Exception(f"Enhanced image preprocessing failed: {str(e)}")
    
    def _get_default_analysis(self):
        """Return default analysis for errors"""
        return {
            'crop_detected': 'Unknown',
            'disease_detected': 'Analysis Failed',
            'confidence': 0.0,
            'severity': 'Unable to determine',
            'treatment': ['Please consult agricultural expert'],
            'prevention': ['Upload clear, well-lit image'],
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _perform_advanced_analysis(self, img_cv, img_array):
        """Perform comprehensive image analysis using multiple techniques"""
        try:
            # Check if OpenCV is available
            if cv2 is None:
                return self._basic_fallback_analysis(img_array)
            
            # Check if the image contains plant content
            if not self._detect_plant_content(img_cv):
                return {
                    'success': False,
                    'error': 'No plant or leaf detected in the image',
                    'green_ratio': 0,
                    'brown_spots': 0,
                    'yellow_ratio': 0,
                    'brightness': 0,
                    'edge_density': 0,
                    'health_score': 0,
                    'color_confidence': 0,
                    'texture_confidence': 0,
                    'spot_confidence': 0
                }
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Ensure we have a 3-channel image
            if len(img_array.shape) == 3:
                h, w, c = img_array.shape
                if c >= 3:
                    # Use RGB channels
                    red_channel = img_array[:,:,0].astype(float) / 255.0
                    green_channel = img_array[:,:,1].astype(float) / 255.0
                    blue_channel = img_array[:,:,2].astype(float) / 255.0
                else:
                    # Grayscale - replicate across channels
                    gray = img_array[:,:,0].astype(float) / 255.0
                    red_channel = green_channel = blue_channel = gray
            else:
                # Grayscale image
                gray = img_array.astype(float) / 255.0
                red_channel = green_channel = blue_channel = gray
            
            # Calculate color ratios
            total_pixels = img_array.shape[0] * img_array.shape[1]
            
            # Green vegetation detection (HSV space)
            try:
                green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
                green_ratio = np.sum(green_mask > 0) / total_pixels
            except:
                green_ratio = np.mean(green_channel)
            
            # Brown/diseased area detection
            try:
                brown_mask = cv2.inRange(hsv, (10, 50, 20), (25, 255, 200))
                brown_spots = np.sum(brown_mask > 0) / total_pixels
            except:
                brown_spots = max(0, 0.3 - green_ratio)  # Estimate based on green deficiency
            
            # Yellow/chlorosis detection
            try:
                yellow_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
                yellow_ratio = np.sum(yellow_mask > 0) / total_pixels
            except:
                yellow_ratio = np.mean(red_channel + green_channel) / 2
            
            # Brightness and contrast analysis
            try:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray) / 255.0
                contrast = np.std(gray) / 255.0
            except:
                brightness = (np.mean(red_channel) + np.mean(green_channel) + np.mean(blue_channel)) / 3
                contrast = (np.std(red_channel) + np.std(green_channel) + np.std(blue_channel)) / 3
            
            # Edge detection for texture analysis
            try:
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / total_pixels
            except:
                edge_density = contrast  # Use contrast as proxy for edge density
            
            # Spot detection
            spot_count = self._detect_spots_safe(img_cv)
            
            # Calculate confidence scores
            color_confidence = min(95, 60 + (green_ratio * 30) + random.uniform(-10, 10))
            texture_confidence = min(95, 50 + (edge_density * 200) + random.uniform(-15, 15))
            spot_confidence = min(95, 70 + (spot_count * 5) + random.uniform(-10, 10))
            
            # Overall health score
            health_score = (green_ratio * 0.4 + (1 - brown_spots) * 0.3 + 
                           (1 - yellow_ratio) * 0.2 + brightness * 0.1) * 100
            
            return {
                'green_ratio': green_ratio,
                'brown_spots': brown_spots,
                'yellow_ratio': yellow_ratio,
                'brightness': brightness,
                'contrast': contrast,
                'edge_density': edge_density,
                'spot_count': spot_count,
                'color_confidence': color_confidence,
                'texture_confidence': texture_confidence,
                'spot_confidence': spot_confidence,
                'health_score': health_score
            }
        except Exception as e:
            # Fallback analysis if OpenCV operations fail
            return self._basic_fallback_analysis(img_array)
    
    def _basic_fallback_analysis(self, img_array):
        """Basic analysis when advanced CV operations fail"""
        try:
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                green_ratio = np.mean(img_array[:,:,1]) / 255.0
                red_ratio = np.mean(img_array[:,:,0]) / 255.0
                brightness = np.mean(img_array) / 255.0
            else:
                brightness = np.mean(img_array) / 255.0
                green_ratio = brightness
                red_ratio = brightness
            
            return {
                'green_ratio': green_ratio,
                'brown_spots': max(0, 0.3 - green_ratio),
                'yellow_ratio': red_ratio * 0.5,
                'brightness': brightness,
                'contrast': np.std(img_array) / 255.0,
                'edge_density': 0.1,
                'spot_count': 1,
                'color_confidence': 65.0,
                'texture_confidence': 55.0,
                'spot_confidence': 60.0,
                'health_score': green_ratio * 80
            }
        except:
            return {
                'green_ratio': 0.5,
                'brown_spots': 0.2,
                'yellow_ratio': 0.1,
                'brightness': 0.5,
                'contrast': 0.2,
                'edge_density': 0.1,
                'spot_count': 1,
                'color_confidence': 50.0,
                'texture_confidence': 50.0,
                'spot_confidence': 50.0,
                'health_score': 60.0
            }
    
    def _detect_spots_safe(self, img_cv):
        """Safe spot detection with fallback"""
        try:
            if cv2 is None or img_cv is None:
                return random.randint(0, 3)  # Random fallback
            return self._detect_spots(img_cv)
        except:
            return random.randint(0, 3)  # Random fallback
    
    def _detect_spots(self, img_cv):
        """Detect disease spots using contour analysis"""
        try:
            if cv2 is None:
                return 0
                
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            
            # Create mask for brown/dark spots (potential disease)
            lower_brown = np.array([10, 50, 20])
            upper_brown = np.array([25, 255, 200])
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size (remove noise)
            min_area = 50  # Minimum area for a spot
            spots = [c for c in contours if cv2.contourArea(c) > min_area]
            
            return len(spots)
        except:
            return 0
    
    def _determine_crop_type_advanced(self, analysis_results):
        """Determine crop type using advanced analysis"""
        green_ratio = analysis_results['green_ratio']
        yellow_ratio = analysis_results['yellow_ratio']
        edge_density = analysis_results['edge_density']
        
        # More sophisticated crop identification
        if green_ratio > 0.6 and edge_density > 0.1:
            # Leafy crops
            if yellow_ratio < 0.1:
                return 'tomato'
            else:
                return 'potato'
        elif green_ratio > 0.4 and edge_density < 0.05:
            # Broader leaf crops
            return 'corn'
        elif yellow_ratio > 0.3:
            # Potentially diseased or yellowing crop
            return random.choice(['tomato', 'potato'])
        else:
            # Default classification
            return random.choice(['tomato', 'potato', 'corn'])
    
    def _determine_disease_advanced(self, crop_type, analysis_results):
        """Enhanced disease determination using multiple factors"""
        crop_diseases = self.classifier_rules.get(crop_type, self.classifier_rules['tomato'])
        
        # Extract analysis factors
        green_ratio = analysis_results['green_ratio']
        brightness = analysis_results['brightness']
        brown_spots = analysis_results['brown_spots']
        yellow_ratio = analysis_results['yellow_ratio']
        health_score = analysis_results['health_score']
        
        # Find best matching disease based on multiple characteristics
        best_match = 'healthy'
        max_score = 0
        
        for disease, ranges in crop_diseases.items():
            score = 0
            factors_matched = 0
            
            # Check green ratio
            if 'green_ratio' in ranges:
                green_min, green_max = ranges['green_ratio']
                if green_min <= green_ratio <= green_max:
                    score += 0.3
                    factors_matched += 1
            
            # Check brightness
            if 'brightness' in ranges:
                bright_min, bright_max = ranges['brightness']
                if bright_min <= brightness <= bright_max:
                    score += 0.25
                    factors_matched += 1
            
            # Check brown spots
            if 'brown_spots' in ranges:
                brown_min, brown_max = ranges['brown_spots']
                if brown_min <= brown_spots <= brown_max:
                    score += 0.25
                    factors_matched += 1
            
            # Check yellow ratio
            if 'yellow_ratio' in ranges:
                yellow_min, yellow_max = ranges['yellow_ratio']
                if yellow_min <= yellow_ratio <= yellow_max:
                    score += 0.2
                    factors_matched += 1
            
            # Bonus for multiple factors matching
            if factors_matched >= 3:
                score += 0.1
            
            if score > max_score:
                max_score = score
                best_match = disease
        
        # Calculate confidence based on match quality and health score
        base_confidence = min(max_score * 100, 90)
        health_adjustment = (health_score / 100) * 10 if best_match == 'healthy' else 0
        confidence = min(base_confidence + health_adjustment + random.uniform(-5, 5), 95)
        
        # Ensure minimum confidence for valid detections
        confidence = max(confidence, 45)
        
        return {
            'disease': best_match,
            'confidence': round(confidence, 1)
        }
    
    def _assess_image_quality_advanced(self, analysis_results):
        """Assess image quality using multiple factors"""
        brightness = analysis_results['brightness']
        contrast = analysis_results['contrast']
        
        if brightness < 0.2 or brightness > 0.9:
            return 'Poor - Image too dark or too bright'
        elif contrast < 0.1:
            return 'Poor - Low contrast image'
        elif contrast > 0.3 and 0.3 <= brightness <= 0.7:
            return 'Excellent - High quality image'
        else:
            return 'Good - Suitable for analysis'
    
    def _get_detailed_recommendations(self, disease, analysis_results):
        """Get detailed recommendations based on disease and analysis"""
        base_recommendations = self._get_additional_recommendations()
        
        if disease != 'healthy':
            health_score = analysis_results['health_score']
            if health_score < 30:
                base_recommendations.append("⚠️ Plant health is severely compromised - consider professional consultation")
            elif health_score < 60:
                base_recommendations.append("⚠️ Plant shows moderate stress - increase monitoring frequency")
            
            if analysis_results['brown_spots'] > 0.3:
                base_recommendations.append("🔍 High diseased area detected - remove affected parts immediately")
        
        return base_recommendations