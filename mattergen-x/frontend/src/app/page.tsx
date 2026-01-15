import HeroSection from "@/components/landing/hero-section";
import Features from "@/components/landing/features-4";
import ContentSection from "@/components/landing/content-7";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <HeroSection />
      <Features />
      <ContentSection />
    </div>
  );
}
