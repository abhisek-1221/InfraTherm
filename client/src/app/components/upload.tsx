"use client"

import React, { useState } from "react"
import { FileUpload } from "@/components/ui/file-upload"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import Image from "next/image"
import { predictImage } from "../../../utils/api"

export function GradCAMVisualization() {
  const [file, setFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [gradcamImage, setGradcamImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleFileUpload = (files: File[]) => {
    if (files.length > 0) {
      setFile(files[0])
      setPrediction(null)
      setGradcamImage(null)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (file) {
      setIsLoading(true)
      try {
        const result = await predictImage(file)
        setPrediction(result.prediction)
        setGradcamImage(`data:image/png;base64,${result.superimposed_img}`)
      } catch (error) {
        console.error('Error predicting image:', error)
      } finally {
        setIsLoading(false)
      }
    }
  }

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Grad-CAM Visualization for Crack Detection</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="w-full min-h-96 border border-dashed bg-white dark:bg-black border-neutral-200 dark:border-neutral-800 rounded-lg">
              <FileUpload onChange={handleFileUpload} />
            </div>
            <div className="flex justify-center items-center">
            <Button type="submit" disabled={!file || isLoading} variant={"destructive"} >
              {isLoading ? "Processing..." : "Predict"}
            </Button>
            </div>
          </form>
        </CardContent>
      </Card>

      {file && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-center">Uploaded Image</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center">
            <Image 
              src={URL.createObjectURL(file)} 
              alt="Uploaded" 
              width={420} 
              height={420} 
              className="rounded-lg"
            />
          </CardContent>
        </Card>
      )}


      {gradcamImage && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-center">Grad-CAM Visualization</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-center">
            <Image 
              src={gradcamImage} 
              alt="Grad-CAM" 
              width={380} 
              height={380} 
              className="rounded-lg"
            />
          </CardContent>
          {prediction && (
                <div className="flex items-center justify-center mb-3">
                    <span className="text-4xl font-extrabold text-blue-300">{prediction}</span>
                </div>
            )}
        </Card>
      )}
    </div>
  )
}
