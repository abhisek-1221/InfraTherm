'use client'

import { motion } from 'framer-motion'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { ModeToggle } from './mode-toggle'

const faqs = [
    {
      question: "What machine learning model is used in this application?",
      answer: "The application uses the EfficientNet architecture, a state-of-the-art Convolutional Neural Network (CNN) known for its efficiency and accuracy, to detect pavement cracks from thermal infrared images."
    },
    {
      question: "How was the model trained and fine-tuned?",
      answer: "The model was pre-trained on the ImageNet dataset and fine-tuned on a custom dataset of thermal infrared images using transfer learning. Hyperparameters such as learning rate, batch size, and number of epochs were optimized using the validation set."
    },
    {
      question: "What performance metrics are used to evaluate the model?",
      answer: "The model’s performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. The model achieved and accuracy of 98.97% and 85.7% on training amd validation sets, respectively."
    },
    {
      question: "What frameworks and technologies are used to build this application?",
      answer: "The front-end is developed using Next.js, the back-end uses FastAPI for efficient API handling, and TensorFlow is employed for building and deploying the machine learning model."
    },
    {
      question: "How does the application handle data pre-processing?",
      answer: "Thermal infrared images are normalized to a [0, 1] range, resized to 224x224 pixels, and augmented using techniques such as rotation and scaling to enhance generalization and prevent overfitting."
    },
    {
      question: "Can the model handle imbalanced datasets effectively?",
      answer: "Yes, techniques such as data augmentation and careful selection of loss functions (e.g., categorical cross-entropy) help address imbalanced datasets, improving the model’s generalization capabilities."
    }
  ]
  
  

export default function FAQ() {
  return (
    <section className="w-full max-w-3xl mx-auto py-12 px-4">
      <motion.h2 
        className="text-3xl font-bold text-center mb-12"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        Frequently Asked Questions
      </motion.h2>
      <Accordion type="single" collapsible className="w-full">
        {faqs.map((faq, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <AccordionItem value={`item-${index}`}>
              <AccordionTrigger>{faq.question}</AccordionTrigger>
              <AccordionContent>{faq.answer}</AccordionContent>
            </AccordionItem>
          </motion.div>
        ))}
      </Accordion>
      <div className='p-4 flex items-center justify-center mt-10'>
        <ModeToggle />
      </div>
    </section>
  )
}

