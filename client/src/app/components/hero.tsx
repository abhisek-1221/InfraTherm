'use client'

import { Cover } from '@/components/ui/cover'
import { motion } from 'framer-motion'
import Image from 'next/image'
import { AnimatedGridPatternDemo } from './animatedpat'
import { CompareDemo } from './autoplay'

export default function Hero() {
  return (
    <section className="w-full max-w-6xl mx-auto text-center py-12 px-4">
      <motion.h1 
        className="text-4xl md:text-6xl font-bold mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        AI-Driven Thermal Imaging Insights with <Cover>InfraTherm</Cover> 
      </motion.h1>
      <motion.p 
        className="text-xl mb-8 text-gray-300"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        for Smarter Infrastructure 

      </motion.p>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className='flex items-center justify-center'
      >
        <CompareDemo />
      </motion.div>
    </section>
  )
}

