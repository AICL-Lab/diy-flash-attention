<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

const props = defineProps<{
  light: string
  dark: string
  alt: string
  caption?: string
}>()

const { isDark } = useData()

const normalizeSrc = (value: string) => {
  if (/^(?:[a-z]+:)?\/\//i.test(value) || value.startsWith('data:')) {
    return value
  }
  return value.startsWith('/') ? withBase(value) : value
}

const src = computed(() => normalizeSrc(isDark.value ? props.dark : props.light))
</script>

<template>
  <figure class="theme-aware-figure">
    <img :src="src" :alt="alt" />
    <figcaption v-if="caption">{{ caption }}</figcaption>
  </figure>
</template>
