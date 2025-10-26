#!/usr/bin/env node
/**
 * Test script to verify Kontext connection and query functionality
 * Usage: node test-kontext.js
 */

require('dotenv').config();

async function testKontextConnection() {
  console.log('=== Kontext Connection Test ===\n');

  // Check environment variables
  const kontextApiKey = process.env.KONTEXT_API_KEY;
  let kontextApiUrl = process.env.KONTEXT_API_URL;
  const kontextUserId = 'd7d3cb73-edf1-46c8-84c4-72703585fdd5';

  // Remove trailing slash from URL if present
  kontextApiUrl = kontextApiUrl ? kontextApiUrl.replace(/\/$/, '') : 'https://api.kontext.dev';

  console.log('Environment Variables:');
  console.log(`- KONTEXT_API_KEY: ${kontextApiKey ? '✓ Set (length: ' + kontextApiKey.length + ')' : '✗ Not set'}`);
  console.log(`- KONTEXT_API_URL: ${kontextApiUrl}`);
  console.log(`- KONTEXT_USER_ID: ${kontextUserId} (hardcoded)\n`);

  if (!kontextApiKey) {
    console.error('❌ Error: KONTEXT_API_KEY must be set in .env file');
    process.exit(1);
  }

  try {
    // Test 1: Initialize Kontext client
    console.log('Test 1: Initializing Kontext client...');
    console.log(`  Using API URL: ${kontextApiUrl}`);
    console.log(`  Using User ID: ${kontextUserId}`);
    
    // Use direct API call with axios instead of SDK
    const axios = require('axios');
    console.log('✓ Kontext client initialized\n');

    // Test 2: List user files
    console.log('Test 2: Listing user files in vault...');
    try {
      const filesResponse = await axios.get(
        `${kontextApiUrl}/v1/vault/files`,
        {
          headers: {
            'Authorization': `Bearer ${kontextApiKey}`,
            'Content-Type': 'application/json',
          },
          params: {
            userId: kontextUserId
          }
        }
      );
      
      const files = filesResponse.data.files || filesResponse.data || [];
      console.log(`✓ Found ${files.length} file(s) in vault:`);
      files.forEach((file, idx) => {
        console.log(`  ${idx + 1}. ${file.fileName || file.name} (ID: ${file.fileId || file.id})`);
      });
      console.log('');
    } catch (error) {
      console.log('⚠️  Could not list files (user may not have any files yet)');
      console.log(`   Error: ${error.message}`);
      if (error.response) {
        console.log(`   Response status: ${error.response.status}`);
        console.log(`   Response data: ${JSON.stringify(error.response.data, null, 2)}`);
      }
      console.log('');
    }

    // Test 3: Query vault with sample procurement question
    console.log('Test 3: Running sample vault query...');
    const testQuery = 'What procurement strategies should I consider for aerospace components?';
    console.log(`Query: "${testQuery}"`);
    
    try {
      const queryResponse = await axios.post(
        `${kontextApiUrl}/v1/vault/query`,
        {
          userId: kontextUserId,
          query: testQuery,
          includeAnswer: true,
        },
        {
          headers: {
            'Authorization': `Bearer ${kontextApiKey}`,
            'Content-Type': 'application/json',
          }
        }
      );

      const result = queryResponse.data;
      console.log('\n✓ Vault query successful\n');
      console.log('Full Response:', JSON.stringify(result, null, 2));
      console.log('\nParsed Response:');
      if (result.answer) {
        console.log(`  Answer: ${result.answer}`);
      } else {
        console.log('  No direct answer provided');
      }

      // Check for facts/results in response
      if (result.facts && result.facts.length > 0) {
        console.log(`\n  Found ${result.facts.length} relevant fact(s):`);
        result.facts.forEach((fact, idx) => {
          console.log(`\n  Fact ${idx + 1}:`);
          if (fact.fileName) console.log(`    File: ${fact.fileName}`);
          if (fact.text) console.log(`    Text: ${fact.text.substring(0, 200)}...`);
        });
      } else {
        console.log('\n  No facts found in response');
      }
      console.log('');
    } catch (error) {
      console.log('⚠️  Vault query failed');
      console.log(`   Error: ${error.message}`);
      if (error.response) {
        console.log(`   Response status: ${error.response.status}`);
        console.log(`   Response data: ${JSON.stringify(error.response.data, null, 2)}`);
      }
      console.log('');
    }

    // Test 4: Get user profile
    console.log('Test 4: Fetching user profile...');
    try {
      const profileResponse = await axios.post(
        `${kontextApiUrl}/v1/profile`,
        {
          userId: kontextUserId,
          task: 'chat',
          userQuery: 'Help with procurement negotiation',
        },
        {
          headers: {
            'Authorization': `Bearer ${kontextApiKey}`,
            'Content-Type': 'application/json',
          }
        }
      );

      const profile = profileResponse.data;
      console.log('✓ Profile retrieved successfully\n');
      console.log('System Prompt Preview:');
      console.log(profile.systemPrompt ? profile.systemPrompt.substring(0, 300) + '...\n' : 'No system prompt available\n');
    } catch (error) {
      console.log('⚠️  Could not fetch profile');
      console.log(`   Error: ${error.message}`);
      if (error.response) {
        console.log(`   Response status: ${error.response.status}`);
        console.log(`   Response data: ${JSON.stringify(error.response.data, null, 2)}`);
      }
      console.log('');
    }

    console.log('=== ✓ All tests completed ===');

  } catch (error) {
    console.error('\n❌ Error during testing:');
    if (error.response) {
      console.error(`  Status: ${error.response.status} ${error.response.statusText}`);
      console.error(`  Response: ${JSON.stringify(error.response.data, null, 2)}`);
    } else {
      console.error(`  ${error.message}`);
    }
    if (error.stack) {
      console.error('\nStack trace:');
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Run the test
testKontextConnection();
