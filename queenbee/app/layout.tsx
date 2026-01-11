import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "BeeUnity | Predictive Beekeeping Intelligence for Makueni",
  description:
    "BeeUnity blends hive acoustics, weather, NDVI, and ward-level context to forecast queen presence, occupancy risk, and honey yield for community beekeepers in Makueni County.",
  keywords: [
    "BeeUnity",
    "Makueni County",
    "beekeeping dashboard",
    "predictive analytics",
    "hive occupancy",
    "queen detection",
  ],
  openGraph: {
    title: "BeeUnity – Predictive Machine Learning for Community Beekeeping",
    description:
      "A data-driven console that unifies queen-state acoustics with climate and NDVI intelligence so Makueni’s beekeepers can act before colony losses.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
