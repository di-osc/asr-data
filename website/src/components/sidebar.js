import React from 'react'
import PropTypes from 'prop-types'
import classNames from 'classnames'

import Link from './link'
import Tag from './tag'
import Dropdown from './dropdown'
import classes from '../styles/sidebar.module.sass'

function getActiveHeading(items, slug) {
    for (let section of items) {
        for (let { isActive, url } of section.items) {
            if (isActive || slug === url) {
                return section.label
            }
        }
    }
    return 'Documentation'
}

const DropdownNavigation = ({ items, defaultValue }) => {
    return (
        <div className={classes['dropdown']}>
            <Dropdown className={classes['dropdown-select']} defaultValue={defaultValue}>
                <option disabled>Select page...</option>
                {items.map((section, i) =>
                    section.items.map(({ text, url }, j) => (
                        <option value={url} key={j}>
                            {section.label} &rsaquo; {text}
                        </option>
                    ))
                )}
            </Dropdown>
        </div>
    )
}

export default function Sidebar({ items = [], slug }) {
    const activeHeading = getActiveHeading(items, slug)

    return (
        <menu className={classNames('sidebar', classes['root'])}>
            <h1 hidden aria-hidden="true" className={classNames('h0', classes['active-heading'])}>
                {activeHeading}
            </h1>
            <DropdownNavigation items={items} defaultValue={slug} />
            {items.map((section, i) => (
                <ul className={classes['section']} key={i}>
                    <li className={classes['label']}>{section.label}</li>
                    {section.items.map(({ text, url, tag, onClick, isActive }, j) => {
                        const active = isActive || slug === url
                        const itemClassNames = classNames(classes['link'], {
                            [classes['is-active']]: active,
                            'is-active': active,
                        })

                        return (
                            <li key={j}>
                                <Link
                                    to={url}
                                    onClick={onClick}
                                    className={itemClassNames}
                                    hideIcon
                                >
                                    {text}
                                    {tag && <Tag spaced>{tag}</Tag>}
                                </Link>
                            </li>
                        )
                    })}
                </ul>
            ))}
        </menu>
    )
}

Sidebar.propTypes = {
    items: PropTypes.arrayOf(
        PropTypes.shape({
            label: PropTypes.string.isRequired,
            items: PropTypes.arrayOf(
                PropTypes.shape({
                    text: PropTypes.string.isRequired,
                    url: PropTypes.string,
                    onClick: PropTypes.func,
                    menu: PropTypes.arrayOf(
                        PropTypes.shape({
                            text: PropTypes.string.isRequired,
                            id: PropTypes.string.isRequired,
                        })
                    ),
                })
            ).isRequired,
        })
    ),
    pageMenu: PropTypes.arrayOf(
        PropTypes.shape({
            text: PropTypes.string.isRequired,
            id: PropTypes.string.isRequired,
        })
    ),
}
