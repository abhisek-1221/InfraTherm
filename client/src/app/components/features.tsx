'use client'

import { motion } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Brain, MonitorCogIcon, BarChart } from 'lucide-react'

const features = [
    {
      icon: Brain, 
      title: "Automated Crack Detection",
      description: "Upload thermal IR imagery to automatically identify and classify pavement cracks, enhancing maintenance efficiency"
    },
    {
      icon: MonitorCogIcon, 
      title: "EfficientNet Architecture",
      description: "Utilize the state-of-the-art EfficientNet model to achieve high accuracy in crack detection while maintaining optimal computational performance"
    },
    {
      icon: BarChart,
      title: "Comprehensive Evaluation",
    description: "Monitor performance metrics and classify detected cracks based on severity and use them for comparative analysis"
    }
  ]
  

export default function Features() {
  return (
    <section className="w-full max-w-6xl mx-auto py-12 px-4">
      <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {features.map((feature, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <Card>
              <CardHeader>
                <feature.icon className="w-10 h-10 mb-4 text-primary" />
                <CardTitle>{feature.title}</CardTitle>
                <CardDescription>{feature.description}</CardDescription>
              </CardHeader>
            </Card>
          </motion.div>
        ))}
      </div>
    </section>
  )
}

