import React from 'react'
import PropTypes from 'prop-types'
import classNames from 'classnames'

import Link from './link'
import Dropdown from './dropdown'
import Icon from './icon'
import classes from '../styles/navigation.module.sass'

const homeUrl = process.env.NEXT_PUBLIC_BASE_PATH || '/'
const repositoryUrl = 'https://github.com/di-osc/asr-data'

const NavigationDropdown = ({ items = [], section }) => {
    const active = items.find(({ url }) => url === '/asr-data')
    return (
        <Dropdown defaultValue={active?.url || 'title'} className={classes.dropdown}>
            <option value="title" disabled>
                文档导航
            </option>
            {items.map(({ text, url }) => (
                <option key={url} value={url}>
                    {text}
                </option>
            ))}
        </Dropdown>
    )
}

export default function Navigation({ title, items = [], section, children }) {
    return (
        <nav className={classes.root}>
            <Link to={homeUrl} aria-label={title} noLinkLayout>
                <span className={classes.title}>
                    {title}
                </span>
            </Link>
            <div className={classes.menu}>
                <NavigationDropdown items={items} section={section} />
                <ul className={classes.list}>
                    {items.map(({ text, url }) => {
                        const isActive = section === text.toLowerCase()
                        return (
                            <li
                                key={url}
                                className={classNames(classes.item, {
                                    [classes['is-active']]: isActive,
                                })}
                            >
                                <Link to={url} tabIndex={isActive ? '-1' : null} noLinkLayout>
                                    {text}
                                </Link>
                            </li>
                        )
                    })}
                </ul>
            </div>
            <Link to={repositoryUrl} className={classes.github} noLinkLayout>
                <Icon name="github" width={18} inline />
                <span>GitHub</span>
            </Link>
            {children}
        </nav>
    )
}

Navigation.propTypes = {
    title: PropTypes.string.isRequired,
    items: PropTypes.array,
    section: PropTypes.string,
    children: PropTypes.node,
}
