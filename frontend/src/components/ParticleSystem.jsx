import React, { useEffect, useRef } from 'react'

const ParticleSystem = () => {
  const containerRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    // Create particles
    const createParticle = () => {
      const particle = document.createElement('div')
      particle.className = 'particle'
      
      // Random starting position
      particle.style.left = Math.random() * 100 + '%'
      particle.style.animationDelay = Math.random() * 20 + 's'
      particle.style.animationDuration = (20 + Math.random() * 10) + 's'
      
      container.appendChild(particle)
      
      // Remove particle after animation
      setTimeout(() => {
        if (container.contains(particle)) {
          container.removeChild(particle)
        }
      }, 30000)
    }

    // Create neural network lines
    const createNeuralLine = () => {
      const line = document.createElement('div')
      line.className = 'neural-line'
      
      // Random position and size
      line.style.top = Math.random() * 100 + '%'
      line.style.left = Math.random() * 100 + '%'
      line.style.width = (50 + Math.random() * 200) + 'px'
      line.style.transform = `rotate(${Math.random() * 360}deg)`
      line.style.animationDelay = Math.random() * 4 + 's'
      
      container.appendChild(line)
      
      // Remove line after some time
      setTimeout(() => {
        if (container.contains(line)) {
          container.removeChild(line)
        }
      }, 8000)
    }

    // Create initial particles and lines
    for (let i = 0; i < 15; i++) {
      setTimeout(createParticle, i * 200)
    }
    
    for (let i = 0; i < 5; i++) {
      setTimeout(createNeuralLine, i * 500)
    }

    // Continuously create new particles and lines
    const particleInterval = setInterval(createParticle, 2000)
    const lineInterval = setInterval(createNeuralLine, 8000)

    return () => {
      clearInterval(particleInterval)
      clearInterval(lineInterval)
    }
  }, [])

  return (
    <div 
      ref={containerRef} 
      className="particle-container neural-network"
      style={{ zIndex: -1 }}
    />
  )
}

export default ParticleSystem
