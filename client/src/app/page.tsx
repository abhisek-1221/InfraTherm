import Image from "next/image";
import Hero from "./components/hero";
import Features from "./components/features";
import FAQ from "./components/faqs";
import AnimatedGridPattern from "@/components/ui/animated-grid-pattern";
import { cn } from "@/lib/utils";
import { FileUploadDemo } from "./components/upload";
import { Button } from "@/components/ui/button";
import { ArrowBigDown } from "lucide-react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
       <AnimatedGridPattern
        numSquares={40}
        maxOpacity={0.1}
        duration={2}
        repeatDelay={1}
        className={cn(
          "[mask-image:radial-gradient(500px_circle_at_center,white,transparent)]",
          "inset-x-0 inset-y-[-30%] h-[200%] skew-y-12",
        )}
      />
    <Hero />
    <Features />
    <Button variant={"secondary"} className="mb-4 py-2 ring-offset-gray-700">
        Try it out ! Upload your image <ArrowBigDown size={24} /> 
    </Button>
    <FileUploadDemo />
    <FAQ />
  </main>
  );
}
