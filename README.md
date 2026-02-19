# PharmaGuard Backend

PharmaGuard is a pharmacogenomics risk assessment API that analyzes patient VCF (Variant Call Format) files against specific drugs to provide clinical recommendations based on CPIC (Clinical Pharmacogenetics Implementation Consortium) guidelines.

## 🚀 Getting Started

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn
- A Google Gemini API Key (for LLM explanations)

### Installation

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   npm install
   ```
3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   PORT=3000
   GEMINI_API_KEY=your_api_key_here
   ```

### Running the Server

- **Development Mode** (with auto-reload):
  ```bash
  npm run dev
  ```
- **Production Build**:
  ```bash
  npm run build
  npm start
  ```

## 🛠 API Documentation

### Analyze VCF for Drug Risk
Analyzes a uploaded VCF file against a list of drugs.

- **URL**: `/api/analyze`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

#### Request Parameters
| Field | Type | Description |
| :--- | :--- | :--- |
| `vcf_file` | File | The `.vcf` file to analyze. |
| `drugs` | String | Comma-separated list of drug names (e.g., `warfarin, clopidogrel`). |

#### Example Request (cURL)
```bash
curl -X POST http://localhost:3000/api/analyze \
  -F "vcf_file=@/path/to/your/patient.vcf" \
  -F "drugs=warfarin,clopidogrel"
```

#### Example Response
```json
{
  "patient_id": "PATIENT_A1B2C3D4",
  "results": [
    {
      "drug": "warfarin",
      "timestamp": "2026-02-19T14:00:00.000Z",
      "risk_assessment": {
        "risk_label": "High Risk",
        "severity": "High"
      },
      "pharmacogenomic_profile": {
        "primary_gene": "VKORC1",
        "diplotype": "*1/*1",
        "phenotype": "Normal Sensitive",
        "detected_variants": [
          { "rsid": "rs9923231", "gene": "VKORC1", "star": "*2" }
        ]
      },
      "clinical_recommendation": {
        "guideline_source": "CPIC",
        "recommendation": "Use a lower starting dose."
      },
      "llm_generated_explanation": {
        "summary": "Patient has a variant that increases sensitivity...",
        "mechanism": "The VKORC1 gene variant affects vitamin K recycling..."
      }
    }
  ]
}
```

## 📁 Project Structure

- `src/index.ts`: Entry point of the application.
- `src/routes/`: Express route definitions.
- `src/services/`: Core logic (VCF parsing, risk calculation, LLM integration).
- `src/rules/`: Genetic rules and clinical guidelines (JSON).
- `uploads/`: Temporary directory for uploaded VCF files.

## 🧪 Technology Stack
- **Framework**: Express.js
- **Runtime**: Node.js (with ESM support)
- **TypeScript**: Typed development for better maintainability.
- **LLM**: Google Gemini API for natural language clinical explanations.
- **VCF Parsing**: Custom readline-based streaming parser.
