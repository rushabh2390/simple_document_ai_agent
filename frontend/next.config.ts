/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone', // Required for multi-stage Docker builds
  typescript: {
    // Optional: set to true if you want builds to succeed even with TS errors
    ignoreBuildErrors: false, 
  },
  eslint: {
    // Optional: set to true if you want builds to succeed even with ESLint errors
    ignoreDuringBuilds: false, 
  },
};

export default nextConfig;