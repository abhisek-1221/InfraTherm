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
    question: "What is AI SaaS?",
    answer: "AI SaaS (Software as a Service) refers to cloud-based software that utilizes artificial intelligence to provide intelligent, automated solutions to businesses and individuals."
  },
  {
    question: "How can AI SaaS benefit my business?",
    answer: "AI SaaS can help streamline your workflows, provide data-driven insights, automate repetitive tasks, and improve decision-making processes, ultimately leading to increased efficiency and productivity."
  },
  {
    question: "Is my data safe with your AI SaaS platform?",
    answer: "Yes, we prioritize data security and privacy. Our platform employs state-of-the-art encryption and follows industry best practices to ensure your data remains safe and confidential."
  },
  {
    question: "Can I integrate your AI SaaS with my existing tools?",
    answer: "Our AI SaaS platform is designed to be highly integrable. We offer APIs and pre-built integrations with many popular tools and platforms to ensure seamless incorporation into your existing workflow."
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

