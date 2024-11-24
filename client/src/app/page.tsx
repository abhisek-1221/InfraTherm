import Image from "next/image";
import Hero from "./components/hero";
import Features from "./components/features";
import FAQ from "./components/faqs";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
    <Hero />
    <Features />
    <FAQ />
  </main>
  );
}
