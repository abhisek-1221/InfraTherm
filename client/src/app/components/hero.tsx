'use client'

import { motion } from 'framer-motion'
import Image from 'next/image'

export default function Hero() {
  return (
    <section className="w-full max-w-6xl mx-auto text-center py-12 px-4">
      <motion.h1 
        className="text-4xl md:text-6xl font-bold mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        AI-Powered SaaS Solution
      </motion.h1>
      <motion.p 
        className="text-xl mb-8 text-gray-600"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        Revolutionize your workflow with cutting-edge AI technology
      </motion.p>
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
      >
        <Image
          src="/hero-image.png"
          alt="AI SaaS Hero Image"
          width={800}
          height={400}
          className="rounded-lg shadow-2xl"
        />
      </motion.div>
    </section>
  )
}

