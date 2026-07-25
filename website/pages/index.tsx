import React from 'react'

import Layout from '../src/templates'
import {
    LandingHeader,
    LandingTitle,
    LandingSubtitle,
    LandingGrid,
    LandingCard,
} from '../src/components/landing'

export default function Home() {
    return (
        <Layout>
            <LandingHeader>
                <LandingTitle>asr-data</LandingTitle>
                <LandingSubtitle>统一管理 ASR 音频、标注与模型预测</LandingSubtitle>
            </LandingHeader>
            <LandingGrid blocks style={undefined}>
                <LandingCard title="ASR-DATA" url="/asr-data" button="查看文档">
                    面向自动语音识别的数据模型、音频处理与 SQLite 存储库，提供 Rust 核心和 Python API。
                </LandingCard>
            </LandingGrid>
        </Layout>
    )
}
