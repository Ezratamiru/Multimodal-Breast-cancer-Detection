"""
Clinical Pathway System - Multi-Modal Breast Cancer Diagnostic Workflow
Provides intelligent test recommendations based on findings from each modality
"""

from typing import Dict, List
from enum import Enum

class Modality(str, Enum):
    XRAY_MAMMOGRAM = "xray_mammogram"
    ULTRASOUND = "ultrasound"
    MRI = "mri"
    BIOPSY = "biopsy"

class ClinicalPathway:
    """Manages clinical decision pathways for breast cancer diagnosis"""
    
    def __init__(self):
        self.modality_info = {
            Modality.XRAY_MAMMOGRAM: {
                "name": "X-ray Mammogram",
                "description": "First-line screening for breast abnormalities",
                "use_cases": [
                    "Routine screening",
                    "Detection of masses and calcifications",
                    "Initial assessment of breast tissue"
                ],
                "strengths": [
                    "Wide availability",
                    "Cost-effective",
                    "Good for detecting calcifications"
                ],
                "limitations": [
                    "Lower sensitivity in dense breasts",
                    "2D projection can miss some lesions"
                ]
            },
            Modality.ULTRASOUND: {
                "name": "Breast Ultrasound",
                "description": "Complementary imaging for characterizing masses",
                "use_cases": [
                    "Distinguishing cystic vs solid masses",
                    "Evaluating palpable lumps",
                    "Supplemental screening in dense breasts"
                ],
                "strengths": [
                    "No radiation exposure",
                    "Excellent for cyst detection",
                    "Real-time imaging"
                ],
                "limitations": [
                    "Operator dependent",
                    "Limited field of view",
                    "Less effective for calcifications"
                ]
            },
            Modality.MRI: {
                "name": "Breast MRI (DCE-MRI)",
                "description": "High-sensitivity imaging for detailed assessment",
                "use_cases": [
                    "High-risk screening",
                    "Evaluating extent of known cancer",
                    "Problem-solving for inconclusive findings"
                ],
                "strengths": [
                    "Highest sensitivity",
                    "3D imaging",
                    "Excellent soft tissue contrast"
                ],
                "limitations": [
                    "Expensive",
                    "Lower specificity (more false positives)",
                    "Requires contrast injection"
                ]
            },
            Modality.BIOPSY: {
                "name": "Tissue Biopsy",
                "description": "Definitive diagnosis through tissue analysis",
                "use_cases": [
                    "Confirming suspicious findings",
                    "Determining benign vs malignant",
                    "Molecular profiling"
                ],
                "strengths": [
                    "Gold standard for diagnosis",
                    "Provides histological information",
                    "Enables treatment planning"
                ],
                "limitations": [
                    "Invasive procedure",
                    "Small risk of complications",
                    "Sampling error possible"
                ]
            }
        }
    
    def recommend_next_tests(self, current_modality: str, findings: Dict) -> Dict:
        """
        Recommend next diagnostic steps based on current findings
        
        Args:
            current_modality: The modality just completed
            findings: Results from the current test
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "current_test": current_modality,
            "findings_summary": self._summarize_findings(current_modality, findings),
            "recommended_tests": [],
            "clinical_pathway": "",
            "urgency": "routine",
            "rationale": []
        }
        
        # X-ray Mammogram results
        if current_modality == Modality.XRAY_MAMMOGRAM:
            if findings.get("prediction") == "Malignant" or findings.get("confidence", 0) > 0.7:
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.ULTRASOUND,
                        "priority": "high",
                        "reason": "Further characterize suspicious mass detected on mammogram"
                    },
                    {
                        "modality": Modality.BIOPSY,
                        "priority": "high",
                        "reason": "Tissue confirmation needed for highly suspicious finding"
                    }
                ]
                recommendations["urgency"] = "urgent"
                recommendations["clinical_pathway"] = "Suspicious → Ultrasound + Biopsy → Histopathology"
                recommendations["rationale"].append("High suspicion finding requires tissue diagnosis")
            else:
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.ULTRASOUND,
                        "priority": "medium",
                        "reason": "Supplemental screening, especially if dense breast tissue"
                    }
                ]
                recommendations["clinical_pathway"] = "Normal/Benign → Consider supplemental ultrasound"
                recommendations["rationale"].append("Low suspicion, routine follow-up appropriate")
        
        # Ultrasound results
        elif current_modality == Modality.ULTRASOUND:
            density = findings.get("density_percentage", 50)
            birads = findings.get("birads_category", "B")
            
            if density > 50 or birads in ["C", "D"]:
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.MRI,
                        "priority": "high",
                        "reason": "Dense breast tissue or heterogeneous findings - MRI for comprehensive assessment"
                    },
                    {
                        "modality": Modality.BIOPSY,
                        "priority": "medium",
                        "reason": "Consider biopsy if discrete lesion identified"
                    }
                ]
                recommendations["urgency"] = "prompt"
                recommendations["clinical_pathway"] = "Dense/Heterogeneous → MRI → Targeted Biopsy if needed"
                recommendations["rationale"].append("Dense tissue limits sensitivity, MRI recommended")
            else:
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.XRAY_MAMMOGRAM,
                        "priority": "routine",
                        "reason": "Annual mammography for routine screening"
                    }
                ]
                recommendations["clinical_pathway"] = "Low density → Routine annual screening"
                recommendations["rationale"].append("Low risk, continue routine surveillance")
        
        # MRI results
        elif current_modality == Modality.MRI:
            risk_level = findings.get("risk_level", "Low")
            
            if risk_level in ["High", "Very High", "Moderate-High"]:
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.BIOPSY,
                        "priority": "urgent",
                        "reason": "MRI shows suspicious enhancement - biopsy required for diagnosis"
                    }
                ]
                recommendations["urgency"] = "urgent"
                recommendations["clinical_pathway"] = "Suspicious MRI → MRI-guided Biopsy → Treatment Planning"
                recommendations["rationale"].append("High-risk MRI finding requires immediate tissue diagnosis")
            elif risk_level == "Moderate":
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.ULTRASOUND,
                        "priority": "medium",
                        "reason": "Second-look ultrasound to correlate MRI findings"
                    },
                    {
                        "modality": Modality.BIOPSY,
                        "priority": "medium",
                        "reason": "Consider biopsy if lesion visible on ultrasound"
                    }
                ]
                recommendations["urgency"] = "prompt"
                recommendations["clinical_pathway"] = "Moderate suspicion → Second-look US → Biopsy if visible"
                recommendations["rationale"].append("Moderate risk requires correlation with other modalities")
            else:
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.XRAY_MAMMOGRAM,
                        "priority": "routine",
                        "reason": "Continue routine screening with annual mammography"
                    }
                ]
                recommendations["clinical_pathway"] = "Low suspicion → Annual screening"
                recommendations["rationale"].append("Low risk MRI, routine follow-up appropriate")
        
        # Biopsy results
        elif current_modality == Modality.BIOPSY:
            prediction = findings.get("prediction", "Benign")
            
            if prediction == "Malignant":
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.MRI,
                        "priority": "urgent",
                        "reason": "Staging MRI to assess extent of disease and guide treatment"
                    }
                ]
                recommendations["urgency"] = "urgent"
                recommendations["clinical_pathway"] = "Malignant Biopsy → Staging MRI → Multidisciplinary Treatment Planning"
                recommendations["rationale"].append("Cancer confirmed - staging and treatment planning required")
            else:
                recommendations["recommended_tests"] = [
                    {
                        "modality": Modality.XRAY_MAMMOGRAM,
                        "priority": "medium",
                        "reason": "Short-interval follow-up mammogram (6 months)"
                    }
                ]
                recommendations["clinical_pathway"] = "Benign Biopsy → 6-month follow-up → Annual screening"
                recommendations["rationale"].append("Benign result, increased surveillance recommended")
        
        return recommendations
    
    def _summarize_findings(self, modality: str, findings: Dict) -> str:
        """Generate human-readable summary of findings"""
        if modality == Modality.XRAY_MAMMOGRAM:
            pred = findings.get("prediction", "Unknown")
            conf = findings.get("confidence", 0) * 100
            return f"{pred} finding with {conf:.1f}% confidence"
        
        elif modality == Modality.ULTRASOUND:
            density = findings.get("density_percentage", 0)
            birads = findings.get("birads_category", "Unknown")
            return f"Breast density {density:.1f}% (BI-RADS {birads})"
        
        elif modality == Modality.MRI:
            pred = findings.get("prediction", "Unknown")
            risk = findings.get("risk_level", "Unknown")
            return f"{pred} with {risk} risk level"
        
        elif modality == Modality.BIOPSY:
            pred = findings.get("prediction", "Unknown")
            conf = findings.get("confidence", 0) * 100
            return f"{pred} diagnosis with {conf:.1f}% confidence"
        
        return "Findings summary not available"
    
    def get_modality_info(self, modality: str) -> Dict:
        """Get detailed information about a specific modality"""
        return self.modality_info.get(modality, {})
    
    def get_complete_workflow(self) -> Dict:
        """Get the complete diagnostic workflow"""
        return {
            "workflow": {
                "step_1": {
                    "modality": "X-ray Mammogram",
                    "purpose": "Initial screening",
                    "typical_findings": ["Normal", "Suspicious mass", "Calcifications"]
                },
                "step_2": {
                    "modality": "Ultrasound",
                    "purpose": "Characterize findings, assess density",
                    "typical_findings": ["Cystic", "Solid", "Dense tissue"]
                },
                "step_3": {
                    "modality": "MRI",
                    "purpose": "High-risk screening or problem-solving",
                    "typical_findings": ["Enhancement patterns", "Multifocal disease"]
                },
                "step_4": {
                    "modality": "Biopsy",
                    "purpose": "Definitive diagnosis",
                    "typical_findings": ["Benign", "Malignant", "Molecular subtype"]
                }
            },
            "clinical_scenarios": {
                "routine_screening": ["Mammogram"],
                "dense_breasts": ["Mammogram", "Ultrasound"],
                "high_risk": ["Mammogram", "MRI"],
                "suspicious_finding": ["Mammogram", "Ultrasound", "Biopsy"],
                "confirmed_cancer": ["Biopsy", "MRI", "Additional staging"]
            }
        }


# Initialize global pathway
clinical_pathway = ClinicalPathway()


if __name__ == "__main__":
    # Example usage
    pathway = ClinicalPathway()
    
    # Scenario 1: Suspicious mammogram
    print("=" * 60)
    print("Scenario 1: Suspicious Mammogram Finding")
    print("=" * 60)
    findings = {"prediction": "Malignant", "confidence": 0.85}
    recs = pathway.recommend_next_tests(Modality.XRAY_MAMMOGRAM, findings)
    print(f"Findings: {recs['findings_summary']}")
    print(f"Urgency: {recs['urgency'].upper()}")
    print(f"Clinical Pathway: {recs['clinical_pathway']}")
    print("\nRecommended Next Tests:")
    for test in recs['recommended_tests']:
        print(f"  - {test['modality']}: {test['reason']} (Priority: {test['priority']})")
    
    print("\n" + "=" * 60)
    print("Scenario 2: Dense Breast on Ultrasound")
    print("=" * 60)
    findings = {"density_percentage": 68, "birads_category": "C"}
    recs = pathway.recommend_next_tests(Modality.ULTRASOUND, findings)
    print(f"Findings: {recs['findings_summary']}")
    print(f"Urgency: {recs['urgency'].upper()}")
    print(f"Clinical Pathway: {recs['clinical_pathway']}")
    print("\nRecommended Next Tests:")
    for test in recs['recommended_tests']:
        print(f"  - {test['modality']}: {test['reason']} (Priority: {test['priority']})")
    
    print("\n" + "=" * 60)
    print("Scenario 3: Malignant Biopsy Result")
    print("=" * 60)
    findings = {"prediction": "Malignant", "confidence": 0.98}
    recs = pathway.recommend_next_tests(Modality.BIOPSY, findings)
    print(f"Findings: {recs['findings_summary']}")
    print(f"Urgency: {recs['urgency'].upper()}")
    print(f"Clinical Pathway: {recs['clinical_pathway']}")
    print("\nRecommended Next Tests:")
    for test in recs['recommended_tests']:
        print(f"  - {test['modality']}: {test['reason']} (Priority: {test['priority']})")
