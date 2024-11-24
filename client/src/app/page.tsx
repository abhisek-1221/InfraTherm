'use client'

import { useState } from 'react'
import { predictImage } from '../../utils/api'
import Image from 'next/image'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [prediction, setPrediction] = useState<string | null>(null)
  const [gradcamImage, setGradcamImage] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (file) {
      try {
        const result = await predictImage(file)
        setPrediction(result.prediction)
        setGradcamImage(`data:image/png;base64,${result.superimposed_img}`)
      } catch (error) {
        console.error('Error predicting image:', error)
      }
    }
  }

  return (
    <div>
      <h1>Grad-CAM Visualization for Crack Detection</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} accept="image/*" />
        <button type="submit">
          Predict
        </button>
      </form>
      {file && (
        <div>
          <h2>Uploaded Image</h2>
          <Image src={URL.createObjectURL(file)} alt="Uploaded" width={224} height={224} />
        </div>
      )}
      {prediction && (
        <div>
          <h2>Prediction: {prediction}</h2>
        </div>
      )}
      {gradcamImage && (
        <div>
          <h2>Grad-CAM Visualization</h2>
          <Image src={gradcamImage} alt="Grad-CAM" width={224} height={224} />
        </div>
      )}
    </div>
  )
}
