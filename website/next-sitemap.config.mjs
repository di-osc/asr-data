const deploymentUrl =
    process.env.SITE_URL ||
    process.env.VERCEL_PROJECT_PRODUCTION_URL ||
    process.env.VERCEL_URL ||
    'http://localhost:3000'
const siteUrl = deploymentUrl.startsWith('http') ? deploymentUrl : `https://${deploymentUrl}`
const isGitHubPages = Boolean(process.env.NEXT_PUBLIC_BASE_PATH)

/** @type {import('next-sitemap').IConfig} */
const config = {
    siteUrl,
    generateRobotsTxt: true,
    autoLastmod: false,
    exclude: isGitHubPages ? ['/'] : [],
}

export default config
