import MDX from '@next/mdx'
import PWA from 'next-pwa'

import remarkPlugins from './plugins/index.mjs'

const withMDX = MDX({
    extension: /\.mdx?$/,
    options: {
        remarkPlugins,
        providerImportSource: '@mdx-js/react',
    },
    experimental: {
        mdxRs: true,
    },
})

const withPWA = PWA({
    dest: 'public',
    disable: process.env.NODE_ENV === 'development',
})

const pagesBasePath = process.env.NEXT_PUBLIC_BASE_PATH || ''
const pagesAssetPrefix = process.env.NEXT_PUBLIC_ASSET_PREFIX || pagesBasePath

/** @type {import('next').NextConfig} */
const nextConfig = withPWA(
    withMDX({
        reactStrictMode: true,
        swcMinify: true,
        pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],
        eslint: {
            ignoreDuringBuilds: true,
        },
        typescript: {
            ignoreBuildErrors: true,
        },
        images: { unoptimized: true },
        assetPrefix: pagesAssetPrefix || undefined,
        trailingSlash: true,
    })
)

export default nextConfig
