"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Send, Upload, X, Image, Plus, Trash2, ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import toast from "react-hot-toast";
import { clinicalApi, streamAnalysis } from "@/api/client";
import type { AnalysisResult, StructuredData, LabValue } from "@/api/client";

interface Props {
  onResult: (result: AnalysisResult) => void;
  onAnalyzing: (v: boolean) => void;
}

interface UploadedImage {
  id: string;
  name: string;
  modality: string;
  size: number;
}

export function ClinicalInputForm({ onResult, onAnalyzing }: Props) {
  const [clinicalText, setClinicalText] = useState("");
  const [uploadedImages, setUploadedImages] = useState<UploadedImage[]>([]);
  const [showStructured, setShowStructured] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadingImage, setUploadingImage] = useState(false);

  // Structured data
  const [age, setAge] = useState<string>("");
  const [sex, setSex] = useState<string>("");
  const [medications, setMedications] = useState<string>("");
  const [allergies, setAllergies] = useState<string>("");
  const [labValues, setLabValues] = useState<Array<{name: string; value: string; unit: string}>>([]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles.slice(0, 3)) {
      setUploadingImage(true);
      try {
        const res = await clinicalApi.uploadImage(file, "other", "");
        setUploadedImages((prev) => [
          ...prev,
          { id: res.image_id, name: file.name, modality: res.modality, size: file.size },
        ]);
        toast.success(`Image "${file.name}" uploaded`);
      } catch (err: any) {
        toast.error(err?.response?.data?.detail || "Image upload failed");
      } finally {
        setUploadingImage(false);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"] },
    maxFiles: 3,
    disabled: isSubmitting,
  });

  const removeImage = (id: string) => {
    setUploadedImages((prev) => prev.filter((img) => img.id !== id));
  };

  const addLabValue = () => {
    setLabValues((prev) => [...prev, { name: "", value: "", unit: "" }]);
  };

  const buildStructuredData = (): StructuredData | undefined => {
    const hasData =
      age || sex || medications || allergies || labValues.some((l) => l.name);
    if (!hasData) return undefined;

    const labsParsed: LabValue[] = labValues
      .filter((l) => l.name && l.value)
      .map((l) => ({
        name: l.name,
        value: parseFloat(l.value) || 0,
        unit: l.unit,
      }));

    return {
      age: age ? parseInt(age) : undefined,
      sex: (sex as any) || undefined,
      medications: medications ? medications.split(",").map((s) => s.trim()).filter(Boolean) : [],
      allergies: allergies ? allergies.split(",").map((s) => s.trim()).filter(Boolean) : [],
      lab_values: labsParsed,
    };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!clinicalText.trim() || clinicalText.length < 10) {
      toast.error("Please provide at least 10 characters of clinical information.");
      return;
    }

    setIsSubmitting(true);
    onAnalyzing(true);
    onResult(null as any);

    try {
      const result = await clinicalApi.analyzeCase({
        clinical_text: clinicalText,
        image_ids: uploadedImages.map((i) => i.id),
        structured_data: buildStructuredData(),
      });
      onResult(result);
      toast.success("Analysis complete");
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Analysis failed. Please try again.";
      toast.error(msg);
    } finally {
      setIsSubmitting(false);
      onAnalyzing(false);
    }
  };

  const charCount = clinicalText.length;
  const charLimit = 5000;

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {/* Clinical Text */}
      <div className="card p-4 space-y-3">
        <label className="block text-sm font-semibold text-clinical-300 uppercase tracking-wider">
          Clinical Notes & Symptoms
        </label>
        <textarea
          className="input-field text-sm leading-relaxed"
          rows={8}
          value={clinicalText}
          onChange={(e) => setClinicalText(e.target.value.slice(0, charLimit))}
          placeholder="Enter clinical observations, symptoms, and patient history...

Example:
65-year-old male, former smoker (40 pack-years). Presents with 3-week history of productive cough, fever (38.8°C), dyspnea on exertion. Physical exam: decreased breath sounds at right base, dullness to percussion. CXR shows right lower lobe opacity."
          disabled={isSubmitting}
        />
        <div className="flex justify-between items-center text-xs text-blue-300/40">
          <span>Provide detailed clinical context for better results</span>
          <span className={charCount > charLimit * 0.9 ? "text-amber-400" : ""}>
            {charCount.toLocaleString()} / {charLimit.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Image Upload */}
      <div className="card p-4 space-y-3">
        <label className="block text-sm font-semibold text-clinical-300 uppercase tracking-wider">
          Medical Images <span className="text-blue-300/40 font-normal normal-case">(optional)</span>
        </label>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-all ${
            isDragActive
              ? "border-clinical-400 bg-clinical-400/10"
              : "border-white/10 hover:border-clinical-500/40 hover:bg-clinical-500/5"
          } ${isSubmitting ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <input {...getInputProps()} />
          {uploadingImage ? (
            <div className="flex items-center justify-center gap-2 text-blue-300/70">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Uploading...</span>
            </div>
          ) : (
            <>
              <Upload className="w-6 h-6 text-blue-300/40 mx-auto mb-2" />
              <p className="text-sm text-blue-300/60">
                {isDragActive ? "Drop images here" : "Drag & drop or click to upload"}
              </p>
              <p className="text-xs text-blue-300/30 mt-1">
                JPEG, PNG, TIFF · Max {Math.round(10)}MB each · Up to 3 images
              </p>
            </>
          )}
        </div>

        {uploadedImages.length > 0 && (
          <div className="space-y-2">
            {uploadedImages.map((img) => (
              <div
                key={img.id}
                className="flex items-center gap-3 p-2.5 rounded-lg bg-clinical-500/10 border border-clinical-500/20"
              >
                <Image className="w-4 h-4 text-clinical-400 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{img.name}</p>
                  <p className="text-xs text-blue-300/50">
                    {img.modality} · {(img.size / 1024).toFixed(0)} KB
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => removeImage(img.id)}
                  className="p-1 rounded hover:bg-rose-500/20 text-blue-300/40 hover:text-rose-400 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Structured Data (collapsible) */}
      <div className="card overflow-hidden">
        <button
          type="button"
          onClick={() => setShowStructured(!showStructured)}
          className="w-full flex items-center justify-between p-4 text-sm font-semibold text-clinical-300 uppercase tracking-wider hover:bg-white/5 transition-colors"
        >
          <span>
            Structured Data{" "}
            <span className="text-blue-300/40 font-normal normal-case">(optional)</span>
          </span>
          {showStructured ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showStructured && (
          <div className="p-4 pt-0 space-y-4 border-t border-white/5">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-blue-300/60 mb-1">Age</label>
                <input
                  type="number"
                  className="input-field text-sm"
                  placeholder="e.g. 45"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  min={0}
                  max={150}
                />
              </div>
              <div>
                <label className="block text-xs text-blue-300/60 mb-1">Sex</label>
                <select
                  className="input-field text-sm"
                  value={sex}
                  onChange={(e) => setSex(e.target.value)}
                >
                  <option value="">Unknown</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-xs text-blue-300/60 mb-1">
                Medications <span className="opacity-50">(comma-separated)</span>
              </label>
              <input
                type="text"
                className="input-field text-sm"
                placeholder="e.g. metformin 500mg, lisinopril 10mg"
                value={medications}
                onChange={(e) => setMedications(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-xs text-blue-300/60 mb-1">
                Allergies <span className="opacity-50">(comma-separated)</span>
              </label>
              <input
                type="text"
                className="input-field text-sm"
                placeholder="e.g. penicillin, sulfa"
                value={allergies}
                onChange={(e) => setAllergies(e.target.value)}
              />
            </div>

            {/* Lab values */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs text-blue-300/60">Lab Values</label>
                <button
                  type="button"
                  onClick={addLabValue}
                  className="flex items-center gap-1 text-xs text-clinical-400 hover:text-clinical-300"
                >
                  <Plus className="w-3.5 h-3.5" /> Add
                </button>
              </div>
              <div className="space-y-2">
                {labValues.map((lab, idx) => (
                  <div key={idx} className="grid grid-cols-[1fr_80px_60px_24px] gap-2 items-center">
                    <input
                      type="text"
                      className="input-field text-sm"
                      placeholder="Name (e.g. WBC)"
                      value={lab.name}
                      onChange={(e) => {
                        const updated = [...labValues];
                        updated[idx].name = e.target.value;
                        setLabValues(updated);
                      }}
                    />
                    <input
                      type="number"
                      className="input-field text-sm"
                      placeholder="Value"
                      value={lab.value}
                      onChange={(e) => {
                        const updated = [...labValues];
                        updated[idx].value = e.target.value;
                        setLabValues(updated);
                      }}
                    />
                    <input
                      type="text"
                      className="input-field text-sm"
                      placeholder="Unit"
                      value={lab.unit}
                      onChange={(e) => {
                        const updated = [...labValues];
                        updated[idx].unit = e.target.value;
                        setLabValues(updated);
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => setLabValues((prev) => prev.filter((_, i) => i !== idx))}
                      className="text-blue-300/30 hover:text-rose-400"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={isSubmitting || !clinicalText.trim() || clinicalText.length < 10}
        className="btn-primary w-full justify-center text-base py-3"
      >
        {isSubmitting ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            Analyzing with RAG + Groq...
          </>
        ) : (
          <>
            <Send className="w-5 h-5" />
            Analyze Clinical Case
          </>
        )}
      </button>
    </form>
  );
}
