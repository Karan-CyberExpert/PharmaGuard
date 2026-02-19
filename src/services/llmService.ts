import { GoogleGenerativeAI } from "@google/generative-ai";
import dotenv from "dotenv";

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || "");

export interface LlmExplanation {
  summary: string;
  mechanism: string;
}

export async function generateExplanation(riskData: any): Promise<LlmExplanation> {
  const model = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

  const prompt = `
You are a clinical pharmacogenomics expert.

Explain the following risk assessment for a healthcare provider.

Drug: ${riskData.drug}
Gene: ${riskData.gene}
Diplotype: ${riskData.diplotype}
Phenotype: ${riskData.phenotype}
Risk Label: ${riskData.risk_label}
Recommendation: ${riskData.recommendation}
Detected Variants: ${JSON.stringify(riskData.detected_variants)}

Return ONLY valid JSON in this exact format:

{
  "summary": "Concise clinical explanation of the risk.",
  "mechanism": "Biological and pharmacokinetic explanation including gene and specific variants."
}

Do not include markdown, headings, or extra commentary.
`;

  try {
    const result = await model.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        responseMimeType: "application/json"
      }
    });

    const response = await result.response;
    const text = response.text();

    const parsed = JSON.parse(text);

    if (!parsed.summary || !parsed.mechanism) {
      throw new Error("Invalid LLM response structure");
    }

    return {
      summary: parsed.summary.trim(),
      mechanism: parsed.mechanism.trim()
    };
  } catch (error) {
    console.warn("LLM generation failed, using fallback.");

    return {
      summary: generateFallbackSummary(riskData),
      mechanism: generateFallbackMechanism(riskData)
    };
  }
}

function generateFallbackSummary(riskData: any): string {
  if (riskData.risk_label === "Safe") {
    return `Patient is a ${riskData.phenotype} for ${riskData.gene}. Standard dosing of ${riskData.drug} is likely appropriate.`;
  }

  if (riskData.risk_label === "Toxic" || riskData.risk_label === "Ineffective") {
    return `Elevated clinical risk identified for ${riskData.drug} due to ${riskData.gene} ${riskData.phenotype} status. ${riskData.recommendation}`;
  }

  return `Dose adjustment or monitoring may be required for ${riskData.drug} based on ${riskData.gene} ${riskData.phenotype} phenotype.`;
}

function generateFallbackMechanism(riskData: any): string {
  const variants = Array.isArray(riskData.detected_variants)
    ? riskData.detected_variants.map((v: any) => v.star).join(", ")
    : "unspecified variants";

  return `The ${riskData.gene} gene (variants: ${variants}) influences metabolism or transporter activity affecting ${riskData.drug} pharmacokinetics, potentially altering drug exposure and response as described in CPIC guidance.`;
}
