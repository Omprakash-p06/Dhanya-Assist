import os
import base64
import io
from PIL import Image
import numpy as np
from datetime import datetime

class VisionService:
    def __init__(self):
        self.supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        
    def analyze_crop_image(self, image_file):
        """Analyze crop image for disease detection"""
        try:
            # Validate image
            validation_result = self._validate_image(image_file)
            if not validation_result['valid']:
                return validation_result
            
            # Process image
            processed_image = self._preprocess_image(image_file)
            
            # For MVP, return mock analysis
            # In production, this would call a trained ML model
            analysis_result = self._mock_disease_detection(processed_image)
            
            return {
                'success': True,
                'analysis': analysis_result,
                'confidence': analysis_result['confidence'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Image analysis failed: {str(e)}',
                'analysis': self._get_default_analysis()
            }
    
    def _validate_image(self, image_file):
        """Validate uploaded image file"""
        try:
            # Check file size
            if hasattr(image_file, 'content_length'):
                if image_file.content_length > self.max_file_size:
                    return {
                        'valid': False,
                        'error': 'File size too large. Maximum 10MB allowed.'
                    }
            
            # Check file format
            filename = getattr(image_file, 'filename', '')
            if filename:
                file_extension = filename.lower().split('.')[-1]
                if file_extension not in self.supported_formats:
                    return {
                        'valid': False,
                        'error': f'Unsupported file format. Supported formats: {", ".join(self.supported_formats)}'
                    }
            
            # Try to open image
            image = Image.open(image_file)
            image.verify()
            
            # Reset file pointer after verification
            image_file.seek(0)
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Invalid image file: {str(e)}'
            }
    
    def _preprocess_image(self, image_file):
        """Preprocess image for analysis"""
        try:
            # Open and resize image
            image = Image.open(image_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to standard size for analysis
            target_size = (224, 224)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize pixel values
            image_array = image_array / 255.0
            
            return {
                'image_array': image_array,
                'original_size': image.size,
                'processed_size': target_size
            }
            
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def _mock_disease_detection(self, processed_image):
        """Mock disease detection for MVP demonstration"""
        # Simulate different disease scenarios
        import random
        
        diseases = [
            {
                'name': 'Leaf Blight',
                'scientific_name': 'Helminthosporium oryzae',
                'severity': 'Moderate',
                'affected_area': '15-25%',
                'treatment': [
                    'Apply copper-based fungicide spray',
                    'Remove affected leaves immediately',
                    'Improve field drainage',
                    'Use resistant varieties for next planting'
                ],
                'prevention': [
                    'Maintain proper plant spacing',
                    'Avoid overhead watering',
                    'Apply preventive fungicide during wet season',
                    'Practice crop rotation'
                ],
                'confidence': round(random.uniform(78, 92), 1)
            },
            {
                'name': 'Rust Disease',
                'scientific_name': 'Puccinia graminis',
                'severity': 'High',
                'affected_area': '25-35%',
                'treatment': [
                    'Apply systemic fungicide immediately',
                    'Remove severely infected plants',
                    'Spray with neem oil solution',
                    'Monitor closely for spread'
                ],
                'prevention': [
                    'Plant rust-resistant varieties',
                    'Ensure good air circulation',
                    'Regular field inspection',
                    'Control alternate hosts nearby'
                ],
                'confidence': round(random.uniform(82, 95), 1)
            },
            {
                'name': 'Healthy Plant',
                'scientific_name': None,
                'severity': 'None',
                'affected_area': '0%',
                'treatment': [
                    'Continue current care routine',
                    'Monitor for any changes',
                    'Maintain proper nutrition',
                    'Ensure adequate water supply'
                ],
                'prevention': [
                    'Regular inspection',
                    'Balanced fertilization',
                    'Proper irrigation management',
                    'Maintain soil health'
                ],
                'confidence': round(random.uniform(85, 98), 1)
            },
            {
                'name': 'Bacterial Wilt',
                'scientific_name': 'Ralstonia solanacearum',
                'severity': 'High',
                'affected_area': '30-40%',
                'treatment': [
                    'Remove and destroy infected plants',
                    'Apply copper-based bactericide',
                    'Improve soil drainage',
                    'Use biocontrol agents'
                ],
                'prevention': [
                    'Use disease-free planting material',
                    'Avoid waterlogged conditions',
                    'Practice crop rotation with non-hosts',
                    'Sanitize farming tools'
                ],
                'confidence': round(random.uniform(75, 88), 1)
            }
        ]
        
        # Select random disease for demonstration
        selected_disease = random.choice(diseases)
        
        # Add additional analysis metadata
        selected_disease.update({
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_quality': 'Good',
            'leaf_condition': self._assess_leaf_condition(),
            'recommendations': self._get_additional_recommendations(selected_disease),
            'market_impact': self._assess_market_impact(selected_disease['severity'])
        })
        
        return selected_disease
    
    def _assess_leaf_condition(self):
        """Assess overall leaf condition"""
        import random
        conditions = [
            'Excellent - Dense green foliage',
            'Good - Minor discoloration visible',
            'Fair - Moderate symptoms present',
            'Poor - Significant damage observed'
        ]
        return random.choice(conditions)
    
    def _get_additional_recommendations(self, disease_info):
        """Get additional farming recommendations"""
        recommendations = []
        
        if disease_info['name'] == 'Healthy Plant':
            recommendations = [
                'Continue monitoring plant health weekly',
                'Maintain current fertilization schedule',
                'Consider preventive organic treatments'
            ]
        else:
            recommendations = [
                'Consult local agricultural extension officer',
                'Document disease progression with photos',
                'Consider soil testing for nutrient deficiency',
                'Join farmer groups for knowledge sharing'
            ]
        
        return recommendations
    
    def _assess_market_impact(self, severity):
        """Assess potential market impact"""
        impact_map = {
            'None': 'No impact - Normal market price expected',
            'Low': 'Minimal impact - 5-10% yield reduction possible',
            'Moderate': 'Moderate impact - 15-25% yield reduction likely',
            'High': 'Significant impact - 25-40% yield reduction expected'
        }
        
        return impact_map.get(severity, 'Impact assessment unavailable')
    
    def _get_default_analysis(self):
        """Return default analysis in case of errors"""
        return {
            'name': 'Analysis Unavailable',
            'scientific_name': None,
            'severity': 'Unknown',
            'affected_area': 'Unable to determine',
            'treatment': ['Please consult local agricultural expert'],
            'prevention': ['Regular monitoring recommended'],
            'confidence': 0.0,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_quality': 'Unknown',
            'leaf_condition': 'Unable to assess',
            'recommendations': ['Upload clear, well-lit image for better analysis'],
            'market_impact': 'Impact assessment unavailable'
        }
    
    def extract_image_metadata(self, image_file):
        """Extract metadata from uploaded image"""
        try:
            image = Image.open(image_file)
            
            return {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'has_transparency': image.mode in ('RGBA', 'LA'),
                'file_size': getattr(image_file, 'content_length', 'Unknown')
            }
            
        except Exception as e:
            return {
                'error': f'Unable to extract metadata: {str(e)}'
            }
    
    def generate_report(self, analysis_result):
        """Generate a detailed report from analysis results"""
        if not analysis_result.get('success', False):
            return "Analysis failed. Please try uploading a different image."
        
        analysis = analysis_result['analysis']
        
        report = f"""
CROP HEALTH ANALYSIS REPORT
Generated on: {analysis['analysis_date']}

DISEASE DETECTION:
Disease: {analysis['name']}
Scientific Name: {analysis.get('scientific_name', 'N/A')}
Severity Level: {analysis['severity']}
Affected Area: {analysis['affected_area']}
Confidence Score: {analysis['confidence']}%

LEAF CONDITION:
{analysis['leaf_condition']}

TREATMENT RECOMMENDATIONS:
{chr(10).join('• ' + treatment for treatment in analysis['treatment'])}

PREVENTION MEASURES:
{chr(10).join('• ' + prevention for prevention in analysis['prevention'])}

ADDITIONAL RECOMMENDATIONS:
{chr(10).join('• ' + rec for rec in analysis['recommendations'])}

MARKET IMPACT ASSESSMENT:
{analysis['market_impact']}

Note: This analysis is based on image recognition technology. For complex cases, 
please consult with local agricultural experts or extension officers.
"""
        return report.strip()
